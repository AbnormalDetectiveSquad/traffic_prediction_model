import threading
import queue
import h5py
import numpy as np
from sklearn import preprocessing
import torch
import geopandas as gpd
import os
import scipy.sparse as sp
import pandas as pd
import tqdm
import time
class DataLoaderContext:
    def __init__(
        self,
        file_manager,
        args,    # FileManager 인스턴스로 변경
        batch_size=16,
        buffer_size=300,
        shuffle=False,
        split_ratio=[0.7175, 0.1825, 0.1],#835 강남 침수  (5월5일 폭우 학습 데이터 : 7월 강남 침수 검증데이터 : 테스트데이터는 평범한 가을)
        mode="train",     # 초기 모드 지정
        Mi_ratio = [0.7,0.3],
    ):
        """
        HDF5 파일에서 데이터를 로드하는 컨텍스트 관리자
        Args:
            file_manager: FileManager 인스턴스
            batch_size (int): 배치 크기
            buffer_size (int): 큐에 유지할 최대 배치 개수
            shuffle (bool): 셔플 여부
            split_ratio (list): [training, validation, test] 비율
            mode (str): 초기 모드 ('train', 'validation', 'test')
        """
        if sum(split_ratio) <= 0:
            raise ValueError("Split ratio must have a positive sum.")
        self.worknum = 0
        self.file_manager = file_manager
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.shuffle = shuffle
        
        # 데이터 분할 계산
        self.split_ratio = [r / sum(split_ratio) for r in split_ratio]
        self.Mi_ratio = [r / sum(Mi_ratio) for r in Mi_ratio]
        train_ratio, val_ratio, test_ratio = self.split_ratio
        
        total_size = self.file_manager.length
        train_end = int(total_size * train_ratio)
        val_end = train_end + int(total_size * val_ratio)
        self.mode=mode
        self.n_his = args.n_his
        self.n_pred = args.n_pred
        self._calculate_ranges(total_size)
        
        # 큐와 스레드 설정
        self.data_queue = queue.Queue(maxsize=buffer_size)
        self.stop_event = threading.Event()
        self.producer_thread = None
        
        # 현재 모드 설정
        self.current_range = self.ranges[mode]
    def reboot(self):
        """컨텍스트 매니저 재초기화"""
        # 기존 스레드/큐 정리
        if self.producer_thread and self.producer_thread.is_alive():
            self.stop_event.set()
            self.producer_thread.join()
        while not self.data_queue.empty():
            self.data_queue.get()
            
        # 상태 초기화
        self.worknum = 0
        self.stop_event.clear()
        self.producer_thread = None
        self.data_queue = queue.Queue(maxsize=self.buffer_size)
    def _calculate_ranges(self, total_size):
        train_ratio, val_ratio, test_ratio = self.split_ratio
        Miratio,_ = self.Mi_ratio
        Mi=int(total_size*Miratio)
        # 각 시퀀스에 필요한 최소 길이
        sequence_length = self.batch_size + self.n_his + self.n_pred - 1
        
        # 나눌 수 있는 최대 시퀀스 수 계산
        usable_size = total_size - sequence_length + 1
        
        # 비율에 맞게 나누기
        train_end = int(usable_size * train_ratio)
        val_end = train_end + int(usable_size * val_ratio)
        
        self.ranges = {
            "train": (0, train_end),
            "validation": (train_end, val_end),
            "test": (val_end, usable_size),
            "full": (0, usable_size),
            "Binary_A":(0,Mi),
            "Binary_B":(Mi,usable_size),
        }

    def set_mode(self, mode):
        """훈련, 검증, 테스트 모드 설정"""
        if mode not in self.ranges:
            raise ValueError("Invalid mode. Choose from ['train', 'validation', 'test','full','Binary_A','Binary_B']")
        self.current_range = self.ranges[mode]

    def _producer(self):
        """데이터를 읽어 큐에 배치를 넣음"""
        start_idx, end_idx = self.current_range
        current_pos = start_idx
        #pbar = tqdm.tqdm(total=end_idx - start_idx, desc='Filling queue')
        while current_pos < end_idx:
            if self.stop_event.is_set():
                break
                
            if (current_pos + self.batch_size > end_idx) and (self.worknum != 0):
                break
                
            # FileManager를 통해 배치 데이터 읽기
            x, y = self.file_manager.read_chunk_training_batch(current_pos)# ddddddddddddddd
            self.data_queue.put((x, y))
            
            current_pos += self.batch_size
            #time.sleep(1e-3)
            self.worknum += 1 
            #pbar.update(self.batch_size)
            #pbar.set_postfix({'queue': f'{self.data_queue.qsize()}/{self.buffer_size}'})
        #pbar.close()  
        self.data_queue.put(None)#         # 에포크 종료 신호
    def __enter__(self):
        """스레드 시작"""
 
        self.producer_thread = threading.Thread(target=self._producer, daemon=True)

        self.producer_thread.start()
        return self.data_queue

    def __exit__(self, exc_type, exc_value, traceback):
        """스레드 종료 및 리소스 정리"""
        self.stop_event.set()
        self.producer_thread.join()
        while not self.data_queue.empty():
            self.data_queue.get()
    @property
    def iterations_per_epoch(self):
        start_idx, end_idx = self.current_range
        total_samples = end_idx - start_idx
        return total_samples // self.batch_size


class BatchFetcher:
    def __init__(self, data_queue,total_iterations=None):
        """
        데이터 큐에서 배치를 가져오는 반복자 클래스
        Args:
            data_queue (queue.Queue): 데이터 큐
        """
        self.data_queue = data_queue
        self.total_iterations = total_iterations
        self.pbar = tqdm.tqdm(total=data_queue.maxsize, desc='Queue status')
        self.current_size = data_queue.qsize()
        self.pbar.n = self.current_size  # 초기 크기로 설정
        self.pbar.refresh()


    def __iter__(self):
        return self

    def __next__(self):
        batch = self.data_queue.get()
        if batch is None:  # 종료 신호
            self.pbar.close()
            raise StopIteration
        
        # 현재 크기 업데이트
        new_size = self.data_queue.qsize()
        self.pbar.n = new_size
        self.pbar.refresh()

        return batch
    def __len__(self):
        if self.total_iterations is None:
            raise NotImplementedError("Total iterations not set")
        return self.total_iterations
    
class DataNormalizer:
   def __init__(self):
       self.zscore = None
   
   def initialize(self, path):
       """저장된 zscore 파라미터를 로드"""
       try:
           loaded = np.load(path)
           self.zscore = preprocessing.StandardScaler()
           self.zscore.mean_ = loaded['mean']
           self.zscore.scale_ = loaded['scale']
           print(f"Loaded z-score parameters from {path}")
           print(f"Mean: {self.zscore.mean_}")
           print(f"Scale: {self.zscore.scale_}")
       except:
           raise FileNotFoundError(f"Could not load normalizer from {path}")
   
   def initialize_from_data(self, data):
       """데이터로부터 zscore 계산 및 정규화된 데이터 반환"""
       self.zscore = preprocessing.StandardScaler()
       normalized_data = self.zscore.fit_transform(data)
       print("Z-score parameters computed from data")
       print(f"Mean: {self.zscore.mean_}")
       print(f"Scale: {self.zscore.scale_}")
       return normalized_data
       
   def generation_file(self, data, path):
       """데이터로부터 zscore 계산하고 파일로 저장"""
       self.zscore = preprocessing.StandardScaler()
       self.zscore.fit(data)
       np.savez(path, 
               mean=self.zscore.mean_,
               scale=self.zscore.scale_)
       print(f"Saved z-score parameters to {path}")
       print(f"Mean: {self.zscore.mean_}")
       print(f"Scale: {self.zscore.scale_}")

   def transform(self, data):
       """데이터 정규화"""
       if self.zscore is None:
           raise ValueError("Normalizer not initialized. Call initialize() or initialize_from_data() first")
       return self.zscore.transform(data)
   
   def inverse_transform(self, data):
       """정규화된 데이터를 원래 스케일로 복원"""
       if self.zscore is None:
           raise ValueError("Normalizer not initialized. Call initialize() or initialize_from_data() first")
       return self.zscore.inverse_transform(data)

class FileManager: 
    def __init__(self, filepath,args,zscore=None):
        self.filepath = filepath
        self.featuresmat = h5py.File(filepath, "r")
        self.final_position = 0
        self.zscore = zscore
        self.length = (self.featuresmat[list(self.featuresmat.keys())[0]]).shape[1]
        self.width = (self.featuresmat[list(self.featuresmat.keys())[0]]).shape[2]
        self.args = args
        self.device = args.device
    def __del__(self,options='all'):
  
        self.featuresmat.close()


    def check_position(self):
        print(f'current position : {self.final_position}')

    def get_key(self):
        print([list(self.featuresmat.keys())[:],list(self.weight.keys())[:]])
    def read_chunk(self, start_pos,end_pos,kind):
        if kind == 'vel':
            data = self.featuresmat[list(self.featuresmat.keys())[0]][start_pos:end_pos][0,:,:]
        elif kind == 'weight':
            data = self.featuresmat[list(self.featuresmat.keys())[0]][start_pos:end_pos][1,:,:]
        else:
            raise ValueError('Invalid kind please choose from [vel, weight]')
        return data
    def read_chunk_training_batch(self, start_pos):
        if self.zscore is None:
            raise ValueError("please input the zscore")
            # 한번에 연속된 범위로 데이터 로드
        end_pos = start_pos + self.args.batch_size + self.args.n_his + self.args.n_pred - 1
        try:
            if not self.featuresmat.id.valid:
                self.featuresmat = h5py.File(self.filepath, "r")
             # features_matrix에서 데이터 읽기
            data = self.featuresmat['features_matrix'][:, start_pos:end_pos, :]  # [특성, 시간, 노드]

            # 배치별로 데이터 만들기
            Result_x = np.zeros([self.args.batch_size, self.args.features, self.args.n_his, self.width])
            Result_y = np.zeros([self.args.batch_size, self.args.n_pred, self.width])
        except Exception as e:
            print(f"Error at position {start_pos}, attempting to reopen files")
        try:
            for i in range(self.args.batch_size):
                # 각 배치의 시작점
                idx = i
                
                # 입력 데이터 준비
                x = data[:,idx:idx + self.args.n_his,:]
                y = data[0,idx + self.args.n_his:idx + self.args.n_his + self.args.n_pred,:]
                
                # 정규화
                x[0,:,:] = self.zscore.transform(x[0,:,:])
                y = self.zscore.transform(y)
                
                # 결과 저장
                Result_x[i,:,:] = x
                Result_y[i] = y
        except Exception as e:
            print(f"Error in batch {i}")
            print(f"y_sol shape: {data.shape if 'data' in locals() else 'not created'}")
            print(f"Result_y shape: {Result_y.shape}")
            raise e
    
        return torch.tensor(Result_x, dtype=torch.float16).to(self.device), torch.tensor(Result_y, dtype=torch.float16).to(self.device)

def delete_if_exists(path):
    if os.path.exists(path):
        if os.path.isfile(path):
            os.remove(path)
            print(f"File '{path}' has been deleted.")
        elif os.path.isdir(path):
            os.rmdir(path)  # Only works for empty directories
            print(f"Directory '{path}' has been deleted.")
        else:
            print(f"'{path}' exists but is neither a file nor an empty directory. Cannot delete.")
    else:
        print(f"'{path}' does not exist.")

def load_adj(arg):  
    dataset_name=arg.dataset
    dataset_path = './data'
    nodes_name="filtered_nodes.shp"
    links_name="filtered_links.shp"
    dataset_path = os.path.join(dataset_path, dataset_name)

    nodes_gdf = gpd.read_file(os.path.join(dataset_path, nodes_name))
    links_gdf = gpd.read_file(os.path.join(dataset_path, links_name))

    save_option, dataset_path_new=check_table_files(dataset_path, nodes_name, links_name)
    if dataset_path_new:  # Ensure it's not None or empty
        delete_if_exists(dataset_path_new)
    dense_matrix,n_links=create_adjacency_matrix(links_gdf, nodes_gdf,True,dataset_path_new,arg.k_threshold)
    adj = sp.csc_matrix(dense_matrix)
    n_vertex = adj.shape[0]
    return adj, n_vertex

def create_adjacency_matrix(links_gdf, nodes_gdf, save_option, dataset_path, k_threshold):
    # Step 1: 길(LINK_ID)을 노드로 간주하고 중심점 계산
    links_gdf['centroid'] = links_gdf.geometry.centroid
    link_ids = sorted(links_gdf['LINK_ID'].unique())
    link_index_map = {link_id: idx for idx, link_id in enumerate(link_ids)}
    n_links = len(link_ids)
    # Step 2: 인접 행렬 초기화
    adj_matrix = np.zeros((n_links, n_links))
    # 모든 거리값을 저장할 리스트
    all_distances = []
    # Step 3: 연결된 링크들 간의 거리 계산
    for _, node in nodes_gdf.iterrows():
        connected_links = links_gdf[
            (links_gdf['F_NODE'] == node['NODE_ID']) | 
            (links_gdf['T_NODE'] == node['NODE_ID'])]
        link_points = connected_links[['LINK_ID', 'centroid']]
        for i, row1 in link_points.iterrows():
            for j, row2 in link_points.iterrows():
                if i < j:
                    dist = row1['centroid'].distance(row2['centroid'])
                    all_distances.append(float(dist))
    #analyze_distance_distribution(all_distances)
    # sigma를 거리의 표준편차로 설정
    sigma = np.std(all_distances)
    # Step 4: 가중치 계산 (k_threshold는 외부에서 받음)
    for _, node in nodes_gdf.iterrows():
        connected_links = links_gdf[
            (links_gdf['F_NODE'] == node['NODE_ID']) | 
            (links_gdf['T_NODE'] == node['NODE_ID'])]
        link_points = connected_links[['LINK_ID', 'centroid']]
        for i, row1 in link_points.iterrows():
            for j, row2 in link_points.iterrows():
                if i < j:
                    idx1 = link_index_map[row1['LINK_ID']]
                    idx2 = link_index_map[row2['LINK_ID']]
                    
                    dist = row1['centroid'].distance(row2['centroid'])
                    
                    if dist <= k_threshold:
                        weight = np.exp(-(dist**2) / (2 * sigma**2))
                        adj_matrix[idx1, idx2] = weight
                        adj_matrix[idx2, idx1] = weight
    print(f"Using manual k_threshold: {k_threshold:.2f}m")
    print(f"Calculated sigma from data: {sigma:.2f}m")
    # Step 5: 정규화
    max_value = np.max(adj_matrix)
    if max_value > 0:
        adj_matrix /= max_value
    if save_option:
        map_df = pd.DataFrame({
            'Matrix_Index': list(link_index_map.values()),
            'Link_ID': list(link_index_map.keys())
        })
        map_df.to_csv(dataset_path, index=False)
        print(f"Index-Link map saved to {dataset_path}")
    sparse_adj_matrix = sp.csr_matrix(adj_matrix)
    sp.save_npz(dataset_path.replace('.csv', '_adj_matrix.npz'), sparse_adj_matrix)
    print(f"Adjacency matrix saved to {dataset_path.replace('.csv', '_adj_matrix.npz')}")
    return adj_matrix, n_links


def check_table_files(dataset_path, nodes_name, links_name):
    # 확장자 제거 후 `_table.csv` 파일 경로 생성
    combined_table_file = os.path.join(
        dataset_path,
        f"{nodes_name.replace('.shp', '')}_{links_name.replace('.shp', '')}_table.csv"
    )
    # 파일 존재 여부 확인
    file_exists = os.path.exists(combined_table_file)
    # save_option 플래그 설정
    save_option = not file_exists
    # 결과 출력
    print(f"Combined table file path: {combined_table_file}")
    print(f"Combined table file exists: {file_exists}")
    print(f"Save option set to: {save_option}")
    return save_option, combined_table_file

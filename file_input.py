import threading
import queue
import h5py
import numpy as np
from sklearn import preprocessing
import torch


class DataLoaderContext:
    def __init__(
        self,
        custom_loader,
        filepath,
        batch_size,
        buffer_size=300,
        shuffle=False,
        split_ratio=[0.7, 0.15, 0.15],
        device="cpu",
    ):
        """
        HDF5 파일에서 데이터를 로드하는 컨텍스트 관리자
        Args:
            custom_loader (callable): 사용자 정의 데이터 로더 함수
            filepath (str): HDF5 파일 경로
            batch_size (int): 배치 크기
            buffer_size (int): 큐에 유지할 최대 배치 개수
            shuffle (bool): 셔플 여부
            split_ratio (list): [training, validation, test] 비율
            device (str): 'cpu' 또는 'cuda'
        """
        if sum(split_ratio) <= 0:
            raise ValueError("Split ratio must have a positive sum.")
        self.custom_loader = custom_loader
        self.filepath = filepath
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.shuffle = shuffle
        self.device = device
        self.split_ratio = [r / sum(split_ratio) for r in split_ratio]  # 비율 정규화
        self.data_queue = queue.Queue(maxsize=buffer_size)
        self.stop_event = threading.Event()
        self.producer_thread = None
        # 데이터셋 길이 확인 및 분할 계산
        self.total_size = self._get_dataset_length()
        self.train_range, self.val_range, self.test_range = self._calculate_splits()

        # 초기 모드는 training
        self.current_range = self.train_range

    def _get_dataset_length(self):
        """HDF5 파일에서 데이터셋 길이를 읽어옴"""
        with h5py.File(self.filepath, "r") as f:
            datasets = list(f.values())
            if not datasets:
                raise ValueError(f"No datasets found in file: {self.filepath}")
            dataset = datasets[0]
            return len(dataset)

    def _calculate_splits(self):
        """[training, validation, test] 비율로 데이터셋 범위를 계산"""
        train_ratio, val_ratio, test_ratio = self.split_ratio
        train_end = int(self.total_size * train_ratio)
        val_end = train_end + int(self.total_size * val_ratio)

        train_range = (0, train_end)
        val_range = (train_end, val_end)
        test_range = (val_end, self.total_size)

        return train_range, val_range, test_range

    def set_mode(self, mode):
        """훈련, 검증, 테스트 모드 설정"""
        if mode == "train":
            self.current_range = self.train_range
        elif mode == "validation":
            self.current_range = self.val_range
        elif mode == "test":
            self.current_range = self.test_range
        else:
            raise ValueError("Invalid mode. Choose from ['train', 'validation', 'test']")

    def _producer(self):
        """custom_loader를 호출하여 데이터를 읽어 큐에 배치를 넣음"""
        start_idx, end_idx = self.current_range
        data_generator = self.custom_loader(
            filepath=self.filepath,
            batch_size=self.batch_size,
            start_idx=start_idx,
            end_idx=end_idx,
            shuffle=self.shuffle,
            device=self.device,
        )

        for batch in data_generator:
            if self.stop_event.is_set():
                break
            self.data_queue.put(batch)

        self.data_queue.put(None)  # 에포크 종료 신호

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

class BatchFetcher:
    def __init__(self, data_queue):
        """
        데이터 큐에서 배치를 가져오는 반복자 클래스
        Args:
            data_queue (queue.Queue): 데이터 큐
        """
        self.data_queue = data_queue

    def __iter__(self):
        return self

    def __next__(self):
        batch = self.data_queue.get()
        if batch is None:  # 종료 신호
            raise StopIteration
        return batch
    
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
    
    def __init__(self, filepath1,filepath2,args,zscore=None):
        self.vel = h5py.File(filepath1, "r")
        self.weight = h5py.File(filepath2, "r")
        self.final_position = 0
        self.zscore = zscore
        self.length = len(self.vel[list(self.vel.keys())[0]])
        self.width = self.vel[list(self.vel.keys())[0]].shape[1]
        self.args = args
        self.device = args.device
    def __del__(self,options='all'):
        if options == 'all':
            self.vel.close()
            self.weight.close()
        elif options == 'vel':
            self.vel.close()
        elif options == 'weight':
            self.weight.close()
        else:
            raise ValueError('Invalid option please choose from [all, vel, weight]')

    def check_position(self):
        print(f'current position : {self.final_position}')

    def get_key(self):
        print([list(self.vel.keys())[:],list(self.weight.keys())[:]])
    def read_chunk(self, start_pos,end_pos,kind):
        if kind == 'vel':
            data = self.vel[list(self.vel.keys())[0]][start_pos:end_pos]
        elif kind == 'weight':
            data = self.weight[list(self.weight.keys())[0]][start_pos:end_pos]
        else:
            raise ValueError('Invalid kind please choose from [vel, weight]')
        return data
    def read_chunk_training_batch_loop(self,start_pos):
        if self.zscore is None:
            raise ValueError("please input the zscore") 
        Result_x = np.zeros([self.args.batch_size, 2, self.args.n_his, self.width])
        Result_y = np.zeros([self.args.batch_size, self.args.n_pred, self.width])
        for i in range(0, self.args.batch_size):
            x = self.vel[list(self.vel.keys())[0]][start_pos:start_pos + self.args.n_his]
            y=self.weight[list(self.weight.keys())[0]][start_pos:start_pos + self.args.n_his]
            sol = self.vel[list(self.vel.keys())[0]][start_pos + self.args.n_his:start_pos + self.args.n_his + self.args.n_pred]
            x = self.zscore.transform(x)
            sol = self.zscore.transform(sol)
            X=np.stack([x, y],axis=0)
            Result_x[i] = X
            Result_y[i] = sol
            start_pos+=1
        return  Result_x, Result_y
    def read_chunk_training_batch(self, start_pos):
        if self.zscore is None:
            raise ValueError("please input the zscore")
        # 한번에 연속된 범위로 데이터 로드
        end_pos = start_pos + self.args.batch_size + self.args.n_his + self.args.n_pred - 1
        data_vel = self.vel[list(self.vel.keys())[0]][start_pos:end_pos]
        data_weight = self.weight[list(self.weight.keys())[0]][start_pos:end_pos]
        
        # 배치별로 데이터 만들기
        Result_x = np.zeros([self.args.batch_size, 2, self.args.n_his, self.width])
        Result_y = np.zeros([self.args.batch_size, self.args.n_pred, self.width])
        
        for i in range(self.args.batch_size):
            # 각 배치의 시작점
            idx = i
            
            # 입력 데이터 준비
            x_vel = data_vel[idx:idx + self.args.n_his]
            x_weight = data_weight[idx:idx + self.args.n_his]
            y_sol = data_vel[idx + self.args.n_his:idx + self.args.n_his + self.args.n_pred]
            
            # 정규화
            x_vel = self.zscore.transform(x_vel)
            y_sol = self.zscore.transform(y_sol)
            
            # 결과 저장
            Result_x[i, 0] = x_vel
            Result_x[i, 1] = x_weight
            Result_y[i] = y_sol
        
        return torch.tensor(Result_x).to(self.device), torch.tensor(Result_y).to(self.device)


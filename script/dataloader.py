import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import geopandas as gpd
def create_adjacency_matrix(links_gdf, nodes_gdf,save_option,dataset_path):
    # Step 1: 길(LINK_ID)을 노드로 간주
    link_ids = links_gdf['LINK_ID'].unique()
    link_index_map = {link_id: idx for idx, link_id in enumerate(link_ids)}
    n_links = len(link_ids)
    
    # Step 2: 인접 행렬 초기화
    adj_matrix = np.zeros((n_links, n_links))
    
    # Step 3: 교차로(NODE_ID)를 기반으로 연결된 길 간 관계 정의
    for _, node in nodes_gdf.iterrows():
        # 교차로에 연결된 모든 길 찾기
        connected_links = links_gdf[
            (links_gdf['F_NODE'] == node['NODE_ID']) | 
            (links_gdf['T_NODE'] == node['NODE_ID'])
        ]['LINK_ID'].unique()
        
        # 연결된 길 간의 관계를 인접 행렬에 추가
        for i in range(len(connected_links)):
            for j in range(i + 1, len(connected_links)):
                idx1 = link_index_map[connected_links[i]]
                idx2 = link_index_map[connected_links[j]]
                adj_matrix[idx1, idx2] = 1
                adj_matrix[idx2, idx1] = 1
    
    # Step 4: 정규화
    max_value = np.max(adj_matrix)
    if max_value > 0:
        adj_matrix /= max_value
        # 매트릭스 인덱스와 링크 ID의 매핑 저장 (save_option이 True인 경우)
    # 매트릭스 인덱스와 링크 ID의 매핑 저장 (save_option이 True인 경우)
    if save_option:
        map_df = pd.DataFrame({
            'Matrix_Index': list(link_index_map.values()),
            'Link_ID': list(link_index_map.keys())
        })
        map_df.to_csv(dataset_path, index=False)
        print(f"Index-Link map saved to {dataset_path}")
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
def load_adj(dataset_name):
    dataset_path = './data'
    if dataset_name != 'seoul':
        dataset_path = os.path.join(dataset_path, dataset_name)
        adj = sp.load_npz(os.path.join(dataset_path, 'adj.npz'))
        adj = adj.tocsc()
        if dataset_name == 'metr-la':
            n_vertex = 207
        elif dataset_name == 'pems-bay':
            n_vertex = 325
        elif dataset_name == 'pemsd7-m':
            n_vertex = 228
    else:
        nodes_name="filtered_nodes.shp"
        links_name="filtered_links.shp"
        dataset_path = os.path.join(dataset_path, dataset_name)
        nodes_gdf = gpd.read_file(os.path.join(dataset_path, nodes_name))
        links_gdf = gpd.read_file(os.path.join(dataset_path, links_name))
        save_option, dataset_path_new=check_table_files(dataset_path, nodes_name, links_name)
        dense_matrix,n_links=create_adjacency_matrix(links_gdf, nodes_gdf,save_option,dataset_path_new)
        adj = sp.csc_matrix(dense_matrix)
        n_vertex = adj.shape[0]
    return adj, n_vertex

def load_data(dataset_name, len_train, len_val):
    dataset_path = './data'
    dataset_path = os.path.join(dataset_path, dataset_name)
    vel = pd.read_csv(os.path.join(dataset_path, 'vel.csv'))
    train = vel[: len_train]
    val = vel[len_train: len_train + len_val]
    test = vel[len_train + len_val:]
    return train, val, test
def data_transform(data, n_his, n_pred, device):
    # produce data slices for x_data and y_data

    n_vertex = data.shape[1]
    len_record = len(data)
    num = len_record - n_his - n_pred
    
    x = np.zeros([num, 1, n_his, n_vertex])
    y = np.zeros([num, n_vertex])
    
    for i in range(num):
        head = i
        tail = i + n_his
        x[i, :, :, :] = data[head: tail].reshape(1, n_his, n_vertex)
        y[i] = data[tail + n_pred - 1]

    return torch.Tensor(x).to(device), torch.Tensor(y).to(device)
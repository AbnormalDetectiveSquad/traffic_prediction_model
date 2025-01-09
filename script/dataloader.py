import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import geopandas as gpd
import matplotlib
matplotlib.use('TkAgg')  # Tk 백엔드 사용
import matplotlib.pyplot as plt


def analyze_distance_distribution(distances, n_bins=30):
    # 도수분포표 계산
    hist, bin_edges = np.histogram(distances, bins=n_bins)
    
    # 각 구간의 범위와 빈도수를 데이터프레임으로 만들기
    intervals = [f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}" for i in range(len(bin_edges)-1)]
    dist_df = pd.DataFrame({
        'Distance Range (m)': intervals,
        'Frequency': hist,
        'Percentage (%)': (hist/len(distances)*100).round(2)
    })
    
    # 누적 비율 추가
    dist_df['Cumulative (%)'] = dist_df['Percentage (%)'].cumsum().round(2)
    
    # 히스토그램 그리기
    plt.figure(figsize=(12, 6))
    plt.hist(distances, bins=n_bins, edgecolor='black')
    plt.title('Distribution of Link Distances')
    plt.xlabel('Distance (m)')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return dist_df
def create_adjacency_matrix(links_gdf, nodes_gdf, save_option, dataset_path, k_threshold):
    # Step 1: 길(LINK_ID)을 노드로 간주하고 중심점 계산
    links_gdf['centroid'] = links_gdf.geometry.centroid
    link_ids = links_gdf['LINK_ID'].unique()
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
            (links_gdf['T_NODE'] == node['NODE_ID'])
        ]
        
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
            (links_gdf['T_NODE'] == node['NODE_ID'])
        ]
        
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
def load_adj(arg):
    dataset_name=arg.dataset
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
        dense_matrix,n_links=create_adjacency_matrix(links_gdf, nodes_gdf,save_option,dataset_path_new,arg.k_threshold)
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
def data_transform(data, n_his, n_pred, device,triple=False):
    # produce data slices for x_data and y_data

    n_vertex = data.shape[1]
    len_record = len(data)
    num = len_record - n_his - n_pred
    
    x = np.zeros([num, 1, n_his, n_vertex])
    if triple:
        y = np.zeros([num, 3, n_vertex])
    else:
        y = np.zeros([num, n_vertex])
    
    for i in range(num):
        head = i
        tail = i + n_his
        x[i, :, :, :] = data[head: tail].reshape(1, n_his, n_vertex)
        if triple:
            y[i,:,:] = data[tail + n_pred - 3:tail + n_pred ]
        else:
            y[i] = data[tail + n_pred - 1]

    return torch.Tensor(x).to(device), torch.Tensor(y).to(device)
import numpy as np
import torch
import scipy.sparse as sp
from torch import nn
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from torch.utils.data import DataLoader,Dataset
import torch_geometric.utils as utils
import math
#######################해당 파트 라이브러리 사용으로 필요 없어짐 ㅠㅠ################################
'''
def chebyshev_polynomials(adj, k):
    """ Chebyshev 다항식 계산 """
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigenval = sp.linalg.eigsh(laplacian, 1, which='LM', return_eigenvectors=False)[0]
    scaled_laplacian = (2. / largest_eigenval) * laplacian - sp.eye(adj.shape[0])

    t_k = []
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k+1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return t_k
def normalize_adj(adj):
    """ 인접 행렬 정규화 """
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
'''
###############################################################################################

class Traffic_prediction_model(torch.nn.Module):            #모델 클래스 정의
    def __init__(self, bottle_channels=16, hidden_channels=64,time_step=12,num_features=5,num_hot_vectors=5,num_nodes=20,batch_size=10):
        super().__init__()
        torch.manual_seed(12345)
        self.conv1 = nn.Conv1d(in_channels=num_nodes, out_channels=(hidden_channels*2), kernel_size=3, padding=0)     #첫번째 컨볼루션 레이어
        self.Gconv1 = GCNConv(in_channels=1, out_channels=hidden_channels)  #GCNConv1(입력 차원, 출력 차원=hidden_channels)
        self.hidden_channels=hidden_channels
        #self.conv1 = nn.Conv1d(in_channels=num_nodes, out_channels=hidden_channels, kernel_size=3, padding=0)  
        #self.conv2 = GCNConv(in_channels=hidden_channels, out_channels=hidden_channels) #GCNConv2(입력 차원=hidden_channels, 출력 차원)

    def forward(self, x, edge_index,hot_vector):
        padding=0
        kernel_size=3
        stride=1
        dilation=1
        batch_size, num_nodes,num_features , time_steps = x.shape
        x = x.permute(0, 1, 3, 2).reshape(batch_size *num_features ,num_nodes, time_steps)
        x=self.conv1(x)
        P = x[:, :x.shape[1] // 2, :]  # 출력의 앞부분 (out_channels)
        Q = x[:, x.shape[1] // 2:, :]  # 출력의 뒷부분 (out_channels)
        x = P * torch.sigmoid(Q)       # GLU 적용
        new_time_steps = math.floor((time_steps + 2*padding - dilation*(kernel_size-1) - 1)/stride + 1)
        x = x.reshape(batch_size, num_features, self.hidden_channels, new_time_steps).permute(0, 1, 3, 2)
        
        batch_size, num_features,num_nodes ,new_time_steps = x.shape
        x = x.permute(0, 3, 2, 1).reshape(batch_size * new_time_steps, num_nodes, num_features) # (batch_size * new_time_steps, num_nodes, self.hidden_channels)
        x = self.Gconv1(x, edge_index[0])
        x = x.reshape(batch_size, new_time_steps, num_nodes, num_features) # 중간 형태 변환
        x = x.permute(0, 3, 2, 1) # 최종 형태 변환
        return x                                            #출력


class Traffic_dataset(Dataset):
    def __init__(self, data):
        if isinstance(data, list):  # 리스트 입력 처리
            processed_data = {'features': [], 'edge_index': [], 'hot_vector': []}
            for i, item in enumerate(data):
                if isinstance(item, dict) and all(k in item for k in ['features', 'edge_index', 'hot_vector']):
                    processed_data['features'].append(item['features'])
                    # 인접 행렬을 edge_index로 변환
                    edge_index = item['edge_index']
                    edge_index = utils.dense_to_sparse(edge_index)[0].long()
                    processed_data['edge_index'].append(edge_index)
                    processed_data['hot_vector'].append(item['hot_vector'])
                else:
                    raise ValueError(f"Dictionary at index {i} is invalid.")
            try:
                self.features = torch.stack(processed_data['features']).clone().detach().to(torch.float16)
                self.edge_index = processed_data['edge_index'][0].clone().detach().long() # edge_index 저장
                self.hot_vector = torch.stack(processed_data['hot_vector']).clone().detach().to(torch.float16)
            except RuntimeError as e:
                if "stack expects a non-empty TensorList" in str(e):
                    raise ValueError("Input list cannot be empty.")
                else:
                    raise RuntimeError(f"Stacking error: {e}")
            self.num_graphs = len(self.features)
            self.num_nodes = self.features.shape[1]
            self.num_features = self.features.shape[2]
            self.hot_vector_dim = self.hot_vector.shape[2]

        elif isinstance(data, dict):  # 딕셔너리 입력 처리
            if not all(k in data for k in ['features', 'edge_index', 'hot_vector']):
                raise KeyError("Data dictionary must contain 'features', 'edge_index', and 'hot_vector' keys.")

            try:
                self.features = data['features'].clone().detach().to(torch.float16)
                # 인접 행렬을 edge_index로 변환
                edge_index = data['edge_index']
                self.edge_index = utils.dense_to_sparse(edge_index)[0].long() # edge_index 저장
                self.hot_vector = data['hot_vector'].clone().detach().to(torch.float16)
            except Exception as e:
                raise TypeError(f"Could not convert data to tensors: {e}")

            if self.features.ndim != 2 and self.features.ndim != 3:
                raise ValueError("Features must be a 2D or 3D tensor.")
            if self.edge_index.ndim != 2: # edge_index 검사 제거
                raise ValueError("edge_index must be a 2D tensor.")
            if self.hot_vector.ndim != 2 and self.hot_vector.ndim != 3:
                raise ValueError("Hot vector must be a 2D or 3D tensor.")

            if self.features.shape[0] != self.hot_vector.shape[0]: # edge_index 검사 제거
                raise ValueError("Number of nodes must be the same for all inputs.")
            
            self.num_graphs = 1
            self.num_nodes = self.features.shape[self.features.ndim-2]
            self.num_features = self.features.shape[self.features.ndim-1]
            self.hot_vector_dim = self.hot_vector.shape[self.hot_vector.ndim-1]
        else:
            raise TypeError("Input data must be a dictionary or a list of dictionaries.")

    def __len__(self):
        return self.num_graphs

    def __getitem__(self, idx):
        if -1 != idx < self.num_graphs:
            features = self.features[idx]
            edge_index = self.edge_index # edge_index 반환
            hot_vector = self.hot_vector[idx]
            return features, edge_index, hot_vector, self.num_features, self.hot_vector_dim
        else:
            raise IndexError("Index out of range.")

  
def generate_random_data(num_nodes, num_features, num_hot_vectors, time_steps):
    """edge_index를 생성하도록 수정."""
    features = torch.randn(time_steps, num_nodes, num_features, dtype=torch.float16).permute(1,2,0)
    # 인접 행렬 생성 (필요한 경우)
    adj_matrix = torch.randint(0, 2, (num_nodes, num_nodes), dtype=torch.float16)
    adj_matrix.fill_diagonal_(1)
    # edge_index 생성
    edge_index = utils.dense_to_sparse(adj_matrix)[0].long()
    assert edge_index.max() < num_nodes, f"Edge index max ({edge_index.max()}) >= num_nodes ({num_nodes})"
    hot_vector = torch.randint(0, 2, (time_steps, num_nodes, num_hot_vectors), dtype=torch.float16).permute(1,2,0)
    return {'features': features, 'edge_index': edge_index, 'hot_vector': hot_vector}
if __name__ == "__main__":                                  #테스트 용 임시 메인 코드 정의 유틸리티파일은 다른 파일에서 불러와 사용
    is_cuda = torch.cuda.is_available()
    print("CUDA Available: ", is_cuda)
    num_nodes = 20
    num_features = 5
    num_hot_vectors = 5
    out_channels = 7
    time_step=12
    dataset=[]
    data = generate_random_data(num_nodes, num_features, num_hot_vectors,time_step)
    Traffic_dataset(data)
    # 랜덤 데이터 생성
    # 랜덤 데이터 생성 (10개 그래프)
    dataset_list = []  # 데이터셋 리스트
    for _ in range(10):
        data= generate_random_data(num_nodes, num_features, num_hot_vectors,time_step)
        dataset_list.append(data) # 각 그래프를 dataset으로 만듭니다.

    # ConcatDataset을 사용하여 모든 데이터셋을 하나로 결합
    dataset = Traffic_dataset(dataset_list)
 
        # 하나의 dataset으로 만듭니다.
    #dataset = Traffic_dataset(dataset_list)
    batch_size=10
    dataloader = DataLoader(dataset, batch_size=batch_size)

    # 모델 생성
    model = Traffic_prediction_model(bottle_channels=16, 
                                     hidden_channels=64,
                                     time_step=time_step,
                                     num_features=num_features,
                                     num_hot_vectors=num_hot_vectors,
                                     num_nodes=num_nodes,
                                     batch_size=batch_size)
    model = model.half()
    for name, param in model.named_parameters():
        print(f"Layer: {name}, Parameter type: {param.dtype}")
    # forward pass (DataLoader 사용)
    for features, edge_index, hot_vector,num_features,hot_vector_dim in dataloader:
        print("Features shape:", features.shape)
        print("edge_index shape:", edge_index.shape)
        print("Hot vector shape:", hot_vector.shape)
        out = model(features, edge_index, hot_vector)
        print(out.shape)

    #print(out)
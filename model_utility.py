import numpy as np
import torch
from torch.utils.data import Dataset
import scipy.sparse as sp
from torch_geometric.nn import GCNConv
import torch.nn.functional as F


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



class Traffic_prediction_model(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

class Traffic_dataset(Dataset):
    def __init__(self,data):
        self.features = data['features']
        self.adjacency = data['adjacency']
        self.hot_vector = data['hot_vector']

    def __len__(self):
        return [len(self.adjacency), len(self.features)]

    def __getitem__(self, idx):
        return self.features[idx], self.adjacency[idx], self.hot_vector[idx]
 
 
 
 
    
if __name__ == "__main__":
    is_cuda = torch.cuda.is_available()
    print("CUDA Available: ", is_cuda)
    # 예시 데이터 생성
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long) # edge index
    x = torch.randn(3, 16)  # 3개의 노드, 각 노드는 16차원 특성 벡터를 가짐

    # 모델 생성
    model = Traffic_prediction_model(in_channels=16, hidden_channels=32, out_channels=7)

    # forward pass
    out = model(x, edge_index)
    print(out)

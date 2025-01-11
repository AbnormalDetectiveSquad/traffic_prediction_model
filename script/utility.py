import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import norm
import torch
import csv

def calc_gso(dir_adj, gso_type):
    n_vertex = dir_adj.shape[0]

    if sp.issparse(dir_adj) == False:
        dir_adj = sp.csc_matrix(dir_adj)
    elif dir_adj.format != 'csc':
        dir_adj = dir_adj.tocsc()

    id = sp.identity(n_vertex, format='csc')

    # Symmetrizing an adjacency matrix
    adj = dir_adj + dir_adj.T.multiply(dir_adj.T > dir_adj) - dir_adj.multiply(dir_adj.T > dir_adj)
    #adj = 0.5 * (dir_adj + dir_adj.transpose())
    
    if gso_type == 'sym_renorm_adj' or gso_type == 'rw_renorm_adj' \
        or gso_type == 'sym_renorm_lap' or gso_type == 'rw_renorm_lap':
        adj = adj + id
    
    if gso_type == 'sym_norm_adj' or gso_type == 'sym_renorm_adj' \
        or gso_type == 'sym_norm_lap' or gso_type == 'sym_renorm_lap':
        row_sum = adj.sum(axis=1).A1
        row_sum_inv_sqrt = np.power(row_sum, -0.5)
        row_sum_inv_sqrt[np.isinf(row_sum_inv_sqrt)] = 0.
        deg_inv_sqrt = sp.diags(row_sum_inv_sqrt, format='csc')
        # A_{sym} = D^{-0.5} * A * D^{-0.5}
        sym_norm_adj = deg_inv_sqrt.dot(adj).dot(deg_inv_sqrt)

        if gso_type == 'sym_norm_lap' or gso_type == 'sym_renorm_lap':
            sym_norm_lap = id - sym_norm_adj
            gso = sym_norm_lap
        else:
            gso = sym_norm_adj

    elif gso_type == 'rw_norm_adj' or gso_type == 'rw_renorm_adj' \
        or gso_type == 'rw_norm_lap' or gso_type == 'rw_renorm_lap':
        row_sum = np.sum(adj, axis=1).A1
        row_sum_inv = np.power(row_sum, -1)
        row_sum_inv[np.isinf(row_sum_inv)] = 0.
        deg_inv = np.diag(row_sum_inv)
        # A_{rw} = D^{-1} * A
        rw_norm_adj = deg_inv.dot(adj)

        if gso_type == 'rw_norm_lap' or gso_type == 'rw_renorm_lap':
            rw_norm_lap = id - rw_norm_adj
            gso = rw_norm_lap
        else:
            gso = rw_norm_adj

    else:
        raise ValueError(f'{gso_type} is not defined.')

    return gso

def calc_chebynet_gso(gso):
    if sp.issparse(gso) == False:
        gso = sp.csc_matrix(gso)
    elif gso.format != 'csc':
        gso = gso.tocsc()

    id = sp.identity(gso.shape[0], format='csc')
    # If you encounter a NotImplementedError, please update your scipy version to 1.10.1 or later.
    eigval_max = norm(gso, 2)

    # If the gso is symmetric or random walk normalized Laplacian,
    # then the maximum eigenvalue is smaller than or equals to 2.
    if eigval_max >= 2:
        gso = gso - id
    else:
        gso = 2 * gso / eigval_max - id

    return gso

def cnv_sparse_mat_to_coo_tensor(sp_mat, device):
    # convert a compressed sparse row (csr) or compressed sparse column (csc) matrix to a hybrid sparse coo tensor
    sp_coo_mat = sp_mat.tocoo()
    i = torch.from_numpy(np.vstack((sp_coo_mat.row, sp_coo_mat.col)))
    v = torch.from_numpy(sp_coo_mat.data)
    s = torch.Size(sp_coo_mat.shape)

    if sp_mat.dtype == np.float32 or sp_mat.dtype == np.float64:
        return torch.sparse_coo_tensor(indices=i, values=v, size=s, dtype=torch.float32, device=device, requires_grad=False)
    else:
        raise TypeError(f'ERROR: The dtype of {sp_mat} is {sp_mat.dtype}, not been applied in implemented models.')
def debug_save(num,x,y,area,zscore,sol=None):
    if sol is not None:
        SS=zscore.inverse_transform(sol[0,0,:].to('cpu').numpy().reshape(1, -1))
        SS=SS.flatten()
        with open(area+f'debugS{num}.csv', mode='w', newline='') as file:
            writer= csv.writer(file)
            writer.writerow([f'S{num}'])
            for q in SS:
                writer.writerow([q])
    YS=zscore.inverse_transform(y[0,0,:].to('cpu').numpy().reshape(1, -1))
    YS=YS.flatten()
    with open(area+f'debugY{num}.csv', mode='w', newline='') as file:
        writer= csv.writer(file)
        writer.writerow([f'Y{num}'])
        for q in YS:
            writer.writerow([q])
    XS=x[0,0,23,:].to('cpu')
    XS=zscore.inverse_transform(XS.detach().numpy().reshape(1, -1))
    XS=XS.flatten()
    with open(area+f'debugX{num}.csv', mode='w', newline='') as file:
        writer= csv.writer(file)
        writer.writerow([f'X{num}'])
        for q in XS:
            writer.writerow([q])
    

def evaluate_model(model, loss, data_iter,args,device,zscore):
    model.eval()
    l_sum, n = 0.0, 0
    with torch.no_grad():
        qq=0
        high=0
        good=0
        with open('Loss.csv', mode='w', newline='') as file:
            writer= csv.writer(file)
            writer.writerow(['Batch_num', 'Loss'])
        for x, y in data_iter:
            if args.graph_conv_type == 'OSA':
                x = x.to(device)
                y = y.to(device)
                y_pred = model(x).squeeze(1)
                l = loss(y_pred, y)

                '''
                if (qq>=0)&(qq<=1000):
                    debug_save(qq,x,y,'bad',zscore,sol=y_pred)
                    high+=1
                if (qq>=300)&(qq<=326):
                    debug_save(qq,x,y,'good',zscore,sol=y_pred)
                    good+=1
                '''
                with open('LossT.csv', mode='a', newline='') as file:
                    writer= csv.writer(file)
                    writer.writerow([qq, l.item()])
                qq+=1
                l_sum += l.item() * (y.numel()/3)  # 배치 평균 손실에 배치 크기를 곱함

                n += (y.numel()/3)  # 총 데이터 개수 누적
            else:# [batch_size, num_nodes, 3]
                y_pred = model(x).view(len(x), -1)  # [batch_size, num_nodes]
                l = loss(y_pred, y)
                l_sum += l.item() * y.shape[0]
                n += y.shape[0]
        mse = l_sum / n
        
        return mse

def evaluate_metric(model, data_iter, scaler, device):
    model.eval()
    with torch.no_grad():
        mae, sum_y, mape, mse = [], [], [], []
        for x, y in data_iter:
            x=x.to(device)
            y=y.to(device)
            y = scaler.inverse_transform(y.view(len(y), -1).cpu().numpy()).reshape(-1)
            y_pred = scaler.inverse_transform(model(x).view(len(x), -1).cpu().numpy()).reshape(-1)
            d = np.abs(y - y_pred)
            mae += d.tolist()
            sum_y += y.tolist()
            mape += (d / (y+1e-10)).tolist()
            mse += (d ** 2).tolist()
        MAE = np.array(mae).mean()
        #MAPE = np.array(mape).mean()
        RMSE = np.sqrt(np.array(mse).mean())
        WMAPE = np.sum(np.array(mae)) / np.sum(np.array(sum_y))

        #return MAE, MAPE, RMSE
        return MAE, RMSE, WMAPE

def evaluate_metric_OSA(model, data_iter, scaler, device):
    model.eval()
    with torch.no_grad():
        mae, sum_y, mape, mse = [], [], [], []
        for x, y in data_iter:
            x=x.to(device)
            y_pred=model(x).squeeze(1).cpu().numpy()
            y=y.numpy()
            for i in range(y_pred.shape[1]):
                y_pred[:,i,:]=scaler.inverse_transform(y_pred[:,i,:])
                y[:,i,:]=scaler.inverse_transform(y[:,i,:])

            y = y.reshape(-1)
            y_pred = y_pred.reshape(-1)
            d = np.abs(y - y_pred)
            mae += d.tolist()
            sum_y += y.tolist()
            mape += (d / (y+1e-10)).tolist()
            mse += (d ** 2).tolist()
        MAE = np.array(mae).mean()
        #MAPE = np.array(mape).mean()
        RMSE = np.sqrt(np.array(mse).mean())
        WMAPE = np.sum(np.array(mae)) / np.sum(np.array(sum_y))

        #return MAE, MAPE, RMSE
        return MAE, RMSE, WMAPE
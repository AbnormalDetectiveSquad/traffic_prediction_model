import logging
import os
import gc
import argparse
import math
import random
import warnings
import tqdm
import numpy as np
import pandas as pd
from sklearn import preprocessing
import csv
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils as utils
import csv
from script import dataloader, utility, earlystopping, opt
from model import models
from torch.amp import autocast, GradScaler
#import nni

def set_env(seed):
    # Set available CUDA devices
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # torch.use_deterministic_algorithms(True)

def get_parameters():
    parser = argparse.ArgumentParser(description='STGCN')
    parser.add_argument('--enable_cuda', type=bool, default=True, help='enable CUDA, default as True')
    parser.add_argument('--seed', type=int, default=42, help='set the random seed for stabilizing experiment results')
    parser.add_argument('--dataset', type=str, default='seoul', choices=['metr-la', 'pems-bay', 'pemsd7-m','seoul'])
    parser.add_argument('--n_his', type=int, default=18)#타임 스텝 1시간이면 12개
    parser.add_argument('--n_pred', type=int, default=6, help='the number of time interval for predcition, default as 3')
    parser.add_argument('--time_intvl', type=int, default=5)
    parser.add_argument('--Kt', type=int, default=3)# Temporal Kernel Size
    parser.add_argument('--stblock_num', type=int, default=4)
    parser.add_argument('--act_func', type=str, default='glu', choices=['glu', 'gtu'])
    parser.add_argument('--Ks', type=int, default=3, choices=[3, 2])
    parser.add_argument('--graph_conv_type', type=str, default='OSA', choices=['cheb_graph_conv', 'graph_conv','OSA'])
    parser.add_argument('--gso_type', type=str, default='sym_norm_lap', choices=['sym_norm_lap', 'rw_norm_lap', 'sym_renorm_adj', 'rw_renorm_adj'])
    parser.add_argument('--enable_bias', type=bool, default=True, help='default as True')
    parser.add_argument('--droprate', type=float, default=0.01)
    parser.add_argument('--lr', type=float, default=0.00015, help='learning rate')
    parser.add_argument('--weight_decay_rate', type=float, default=0.00001, help='weight decay (L2 penalty)')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=1000, help='epochs, default as 1000')
    parser.add_argument('--opt', type=str, default='adamw', choices=['adamw', 'nadamw', 'lion'], help='optimizer, default as nadamw')
    parser.add_argument('--step_size', type=int, default=18)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
    parser.add_argument('--complexity', type=int, default=8, help='number of bottleneck chnnal | in paper value is 16')
    parser.add_argument('--k_threshold', type=float, default=250.0, help='adjacency_matrix threshold parameter menual setting')
    parser.add_argument('--fname', type=str, default='30_step_8_base_0.0015_gangnam', help='name')
    args = parser.parse_args()
    print('Training configs: {}'.format(args))

    # For stable experiment results
    set_env(args.seed)

    # Running in Nvidia GPU (CUDA) or CPU
    if args.enable_cuda and torch.cuda.is_available():
        # Set available CUDA devices
        # This option is crucial for multiple GPUs
        # 'cuda' ≡ 'cuda:0'
        device = torch.device('cuda')
        torch.cuda.empty_cache() # Clean cache
    else:
        device = torch.device('cpu')
        gc.collect() # Clean cache
    
    Ko = args.n_his - (args.Kt - 1) * 2 * args.stblock_num

    # blocks: settings of channel size in st_conv_blocks and output layer,
    # using the bottleneck design in st_conv_blocks
    blocks = []
    blocks.append([1])
    n=args.complexity
    for l in range(args.stblock_num):
        blocks.append([int(n*2), int(n), int(n*2)])
    if Ko == 0:
        blocks.append([int(n*8)])
    elif Ko > 0:
        blocks.append([int(n*8), int(n*8)])
    blocks.append([1])
    
    return args, device, blocks

def data_preparate(args, device):    
    adj, n_vertex = dataloader.load_adj(args)
    gso = utility.calc_gso(adj, args.gso_type)
    if args.graph_conv_type == 'cheb_graph_conv' or 'OSA':
        gso = utility.calc_chebynet_gso(gso)
    gso = gso.toarray()
    gso = gso.astype(dtype=np.float32)
    args.gso = torch.from_numpy(gso).to(device)
    #args.dataset = 'metr-la'
    dataset_path = './data'
   # if args.dataset != 'seoul':
    dataset_path = os.path.join(dataset_path, args.dataset)
    data_col = pd.read_csv(os.path.join(dataset_path, 'vel.csv')).shape[0]
    # recommended dataset split rate as train: val: test = 60: 20: 20, 70: 15: 15 or 80: 10: 10
    # using dataset split rate as train: val: test = 70: 15: 15
    val_and_test_rate = 0.15
    len_val = int(math.floor(data_col * val_and_test_rate))
    len_test = int(math.floor(data_col * val_and_test_rate))
    len_train = int(data_col - len_val - len_test)
    #else:


    train, val, test = dataloader.load_data(args.dataset, len_train, len_val)
    zscore = preprocessing.StandardScaler()
    train = zscore.fit_transform(train)
    val = zscore.transform(val)
    test = zscore.transform(test)

    x_train, y_train = dataloader.data_transform(train, args.n_his, args.n_pred, device)
    x_val, y_val = dataloader.data_transform(val, args.n_his, args.n_pred, device)
    x_test, y_test = dataloader.data_transform(test, args.n_his, args.n_pred, device)

    train_data = utils.data.TensorDataset(x_train, y_train)
    train_iter = utils.data.DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=False)
    val_data = utils.data.TensorDataset(x_val, y_val)
    val_iter = utils.data.DataLoader(dataset=val_data, batch_size=args.batch_size, shuffle=False)
    test_data = utils.data.TensorDataset(x_test, y_test)
    test_iter = utils.data.DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=False)

    return n_vertex, zscore, train_iter, val_iter, test_iter

def prepare_model(args, blocks, n_vertex):
    loss = nn.MSELoss()
    es = earlystopping.EarlyStopping(delta=0.0, 
                                     patience=args.patience, 
                                     verbose=True, 
                                     path="STGCN_" + args.dataset + args.fname + ".pt")

    if args.graph_conv_type == 'cheb_graph_conv':
        model = models.STGCNChebGraphConv(args, blocks, n_vertex).to(device)
    elif args.graph_conv_type == 'OSA':
        model = models.STGCNChebGraphConv_OSA(args, blocks, n_vertex).to(device)
    else:
        model = models.STGCNGraphConv(args, blocks, n_vertex).to(device)

    if args.opt == "adamw":
        optimizer = optim.AdamW(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay_rate)
    elif args.opt == "nadamw":
        optimizer = optim.NAdam(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay_rate, decoupled_weight_decay=True)
    elif args.opt == "lion":
        optimizer = opt.Lion(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay_rate)
    else:
        raise ValueError(f'ERROR: The {args.opt} optimizer is undefined.')

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    return loss, es, model, optimizer, scheduler

def train(args, model, loss, optimizer, scheduler, es, train_iter, val_iter):
        # CSV 파일을 "쓰기 모드"로 열고, 필요하다면 헤더를 기록
    # - 'w'를 쓰면 매번 덮어씌워집니다. 이미 파일이 있으면 'a'로 열어 이어쓰기 가능
    csv_path=f"train_log_{args.dataset+args.fname}.csv"
    if not os.path.exists(csv_path):
        with open(f"train_log_{args.dataset+args.fname}.csv", mode="w", newline="") as f:# CSV 헤더 한 번 작성 (원하면 생략 가능)
            writer = csv.writer(f)
            writer.writerow(["Epoch", "LR", "TrainLoss", "ValLoss", "GPUMem(MB)"])
    else:
        with open(f"train_log_{args.dataset+args.fname}.csv", mode="a", newline="") as f:# CSV 헤더 한 번 작성 (원하면 생략 가능)
            writer = csv.writer(f)
            writer.writerow(["New Epoch", "LR", "TrainLoss", "ValLoss", "GPUMem(MB)"])
    for epoch in range(args.epochs):
        l_sum, n = 0.0, 0  # 'l_sum' is epoch sum loss, 'n' is epoch instance number
        model.train()
        scaler = GradScaler()
        for x, y in tqdm.tqdm(train_iter):
            
            optimizer.zero_grad()
            with autocast(device_type='cuda', dtype=torch.float16):
                y_pred = model(x).view(len(x), -1)  # [batch_size, num_nodes]
                l = loss(y_pred, y)
            scaler.scale(l).backward()
            scaler.step(optimizer)
            scaler.update()
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]
        scheduler.step()
        val_loss = val(model, val_iter)
        # GPU memory usage
        gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1000000 if torch.cuda.is_available() else 0
        print('Epoch: {:03d} | Lr: {:.20f} |Train loss: {:.6f} | Val loss: {:.6f} | GPU occupy: {:.6f} MiB'.\
            format(epoch+1, optimizer.param_groups[0]['lr'], l_sum / n, val_loss, gpu_mem_alloc))
        # **CSV에도 기록**: [에폭, LR, 훈련손실, 검증손실, GPU사용량]
        with open(csv_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch+1,
                optimizer.param_groups[0]['lr'],
                f"{l_sum / n:.6f}",
                f"{val_loss:.6f}",
                f"{gpu_mem_alloc:.2f}"
            ])
            print("csv data saved")
        es(val_loss, model)
        if es.early_stop:
            print("Early stopping")
            break

@torch.no_grad()
def val(model, val_iter):
    model.eval()

    l_sum, n = 0.0, 0
    for x, y in val_iter:
        y_pred = model(x).view(len(x), -1)
        l = loss(y_pred, y)
        l_sum += l.item() * y.shape[0]
        n += y.shape[0]
    return torch.tensor(l_sum / n)


@torch.no_grad() 
def test(zscore, loss, model, test_iter, args):
    model.load_state_dict(torch.load("STGCN_" + args.dataset + args.fname + ".pt"))
    model.eval()

    test_MSE = utility.evaluate_model(model, loss, test_iter)
    test_MAE, test_RMSE, test_WMAPE = utility.evaluate_metric(model, test_iter, zscore)
    print(f'Dataset {args.dataset:s} | Test loss {test_MSE:.6f} | MAE {test_MAE:.6f} | RMSE {test_RMSE:.6f} | WMAPE {test_WMAPE:.8f}')
def test2(zscore, loss, model, test_iter, args):
    # Load the model weights
    model.load_state_dict(torch.load("STGCN_" + args.dataset + args.fname + ".pt"))
    model.eval()

    # Evaluate the model
    test_MSE = utility.evaluate_model(model, loss, test_iter)
    test_MAE, test_RMSE, test_WMAPE = utility.evaluate_metric(model, test_iter, zscore)

    # Prepare CSV output
    output_file = f"test_results_{args.dataset+args.fname}.csv"
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Ground Truth", "Prediction"])

        max_batches = 16
        mid_point = args.batch_size // 2
        
        for i, batch in enumerate(test_iter):
            if i >= max_batches:
                break

            inputs, ground_truth = batch
            inputs, ground_truth = inputs.to(device), ground_truth.to(device)

            with torch.no_grad():
                predictions = model(inputs).view(len(inputs), -1)

            # Denormalize if needed (apply zscore reverse transformation)
            ground_truth = zscore.inverse_transform(ground_truth.cpu().numpy())
            predictions = zscore.inverse_transform(predictions.cpu().numpy())

            # 각 배치에서 첫 번째와 중간 위치의 데이터 저장
            indices = [0, mid_point]
            for idx in indices:
                gt_slice = ground_truth[idx]
                pred_slice = predictions[idx]
                for gt, pred in zip(gt_slice, pred_slice):
                    writer.writerow([f"{gt:.6f}", f"{pred:.6f}"])

    print(f'Dataset {args.dataset:s} | Test loss {test_MSE:.6f} | MAE {test_MAE:.6f} | RMSE {test_RMSE:.6f} | WMAPE {test_WMAPE:.8f}')
    print(f"Test results saved to {output_file}")
    
    
if __name__ == "__main__":
    # Logging
    #logger = logging.getLogger('stgcn')
    #logging.basicConfig(filename='stgcn.log', level=logging.INFO)
    logging.basicConfig(level=logging.INFO)

    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    args, device, blocks = get_parameters()
    n_vertex, zscore, train_iter, val_iter, test_iter = data_preparate(args, device)
    loss, es, model, optimizer, scheduler = prepare_model(args, blocks, n_vertex)
    train(args, model, loss, optimizer, scheduler, es, train_iter, val_iter)# 모델평가만 원할때 추석처리
    test2(zscore, loss, model, test_iter, args) # 모델 평가시 csv로 ground truth 와 prediction 저장 원할 시 사용
    #test(zscore, loss, model, test_iter, args) # 평가 결과면 원할 시 사용

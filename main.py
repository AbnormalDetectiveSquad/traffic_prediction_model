import logging
import os
import gc
import argparse
import random
import warnings
import tqdm
import numpy as np
import csv
import torch
import torch.nn as nn
import torch.optim as optim
import csv
from script import file_input as fi
from script import utility, earlystopping, opt
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
    parser.add_argument('--n_his', type=int, default=24)#타임 스텝 1시간이면 12개


    parser.add_argument('--n_pred', type=int, default=3, help='the number of time interval for predcition, default as 3')

    parser.add_argument('--time_intvl', type=int, default=5)

    parser.add_argument('--Kt', type=int, default=3)# Temporal Kernel Size
    parser.add_argument('--stblock_num', type=int, default=5)
    parser.add_argument('--act_func', type=str, default='glu', choices=['glu', 'gtu'])
    parser.add_argument('--Ks', type=int, default=3, choices=[3, 2])
    parser.add_argument('--graph_conv_type', type=str, default='OSA', choices=['cheb_graph_conv', 'graph_conv','OSA'])
    parser.add_argument('--gso_type', type=str, default='sym_norm_lap', choices=['sym_norm_lap', 'rw_norm_lap', 'sym_renorm_adj', 'rw_renorm_adj'])
    parser.add_argument('--enable_bias', type=bool, default=True, help='default as True')
    
    
    parser.add_argument('--droprate', type=float, default=0.01)


    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    


    parser.add_argument('--batch_size', type=int, default=16)



    parser.add_argument('--weight_decay_rate', type=float, default=0.00000, help='weight decay (L2 penalty)')
    
    
    
    parser.add_argument('--epochs', type=int, default=1000, help='epochs, default as 1000')
    parser.add_argument('--opt', type=str, default='adamw', choices=['adamw', 'nadamw', 'lion'], help='optimizer, default as nadamw')
    parser.add_argument('--step_size', type=int, default=18)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
    parser.add_argument('--k_threshold', type=float, default=460.0, help='adjacency_matrix threshold parameter menual setting')


    parser.add_argument('--complexity', type=int, default=16, help='number of bottleneck chnnal | in paper value is 16')
  

    parser.add_argument('--fname', type=str, default='K460_16base_S220samp_seq_lr0.0001_0112p', help='name')
    parser.add_argument('--mode', type=str, default='test', help='test or train')
    parser.add_argument('--HotEncoding', type=str, default="On", help='On or Off')
    parser.add_argument('--Continue', type=str, default="False", help='True or False')
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
    args.device = device
    Ko = args.n_his - (args.Kt - 1) * 2 * args.stblock_num

    # blocks: settings of channel size in st_conv_blocks and output layer,
    # using the bottleneck design in st_conv_blocks
    blocks = []
    if args.HotEncoding == 'On':
        blocks.append([2])
    else:
        blocks.append([1])

    n=args.complexity
    for l in range(args.stblock_num):
        blocks.append([int(n*4), int(n), int(n*4)])
    if Ko == 0:
        blocks.append([int(n*8)])
    elif Ko > 0:
        blocks.append([int(n*8), int(n*8)])
    blocks.append([1])
    
    return args, device, blocks

def setup_preprocess(args,file_path1,file_path2):
    adj, n_vertex = fi.load_adj(args)
    gso = utility.calc_gso(adj, args.gso_type)
    gso = utility.calc_chebynet_gso(gso)
    gso = gso.toarray()
    gso = gso.astype(dtype=np.float32)
    args.gso = torch.from_numpy(gso).to(device)
    gso=None
    dataset_path = './data'
    dataset_path = os.path.join(dataset_path, args.dataset)
    data = fi.FileManager(file_path1,file_path2,args)
    zscore = fi.DataNormalizer()
    vel = data.read_chunk(0, 5000, 'vel')
    zscore.initialize_from_data(vel)
    data.zscore = zscore
    x, sol = data.read_chunk_training_batch(0)
    train_iter=fi.DataLoaderContext(
        data,    # FileManager 인스턴스로 변경
        args,
        batch_size=args.batch_size,
        buffer_size=300,
        shuffle=False,
        split_ratio=[0.7, 0.15, 0.15],
        mode="train"     
    )
    val_iter=fi.DataLoaderContext(
        data,
        args,    # FileManager 인스턴스로 변경
        batch_size=args.batch_size,
        buffer_size=300,
        shuffle=False,
        split_ratio=[0.7, 0.15, 0.15],
        mode="validation"     
    )
    test_iter=fi.DataLoaderContext(
        data,    # FileManager 인스턴스로 변경
        args,
        batch_size=args.batch_size,
        buffer_size=300,
        shuffle=False,
        split_ratio=[0.7, 0.15, 0.15],
        mode="test"     
    )
    print(x[0,0,0,0],sol[0,0,0])
    return data, args, n_vertex, zscore,train_iter,val_iter,test_iter


def setup_model(args, blocks, n_vertex):
    loss = nn.MSELoss()
    es = earlystopping.EarlyStopping(delta=0.0, 
                                     patience=args.patience, 
                                     verbose=True, 
                                     path="./Weight/STGCN_" + args.dataset + args.fname + ".pt")
    start_epoch = 0
    best_val_loss = float('inf')
    
    if args.Continue == "True":
        model = models.STGCNChebGraphConv_OSA(args, blocks, n_vertex).to(device)
        checkpoint_path = "./Weight/STGCN_" + args.dataset + args.fname + ".pt"
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])

        print(f"Loaded weights from {checkpoint_path}")
    else:
        checkpoint=None
        model = models.STGCNChebGraphConv_OSA(args, blocks, n_vertex).to(device)
    if args.opt == "adamw":
        optimizer = optim.AdamW(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay_rate)
    elif args.opt == "nadamw":
        optimizer = optim.NAdam(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay_rate, decoupled_weight_decay=True)
    elif args.opt == "lion":
        optimizer = opt.Lion(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay_rate)
    else:
        raise ValueError(f'ERROR: The {args.opt} optimizer is undefined.')
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        best_val_loss = checkpoint['val_loss']
    es.val_loss_min = best_val_loss
    return loss, es, model, optimizer, scheduler ,start_epoch, best_val_loss

@torch.no_grad()
def val(model, val_iter):
    model.eval()
    l_sum, n = 0.0, 0
    #qq=0
    with val_iter as queue:
        batch_fetcher = fi.BatchFetcher(queue)
        for x, y in tqdm.tqdm(batch_fetcher, total=val_iter.iterations_per_epoch):
            with autocast(device_type='cuda', dtype=torch.float16):
                y_pred = model(x).squeeze(1)
                l = loss(y_pred, y)
            l_sum += l.item() * y.shape[0]
            #qq+=1
            n += y.shape[0]
    return torch.tensor(l_sum / n)

def train(args, model, loss, optimizer, scheduler, es, train_iter, val_iter,start_epoch, best_val_loss):
        # CSV 파일을 "쓰기 모드"로 열고, 필요하다면 헤더를 기록
    # - 'w'를 쓰면 매번 덮어씌워집니다. 이미 파일이 있으면 'a'로 열어 이어쓰기 가능
    csv_path=f"./Log/train_log_{args.dataset+args.fname}.csv"
    if not os.path.exists(csv_path):
        with open(csv_path, mode="w", newline="") as f:# CSV 헤더 한 번 작성 (원하면 생략 가능)
            writer = csv.writer(f)
            writer.writerow(["Epoch", "LR", "TrainLoss", "ValLoss", "GPUMem(MB)"])
    else:
        with open(csv_path, mode="a", newline="") as f:# CSV 헤더 한 번 작성 (원하면 생략 가능)
            writer = csv.writer(f)
            writer.writerow(["New Epoch", "LR", "TrainLoss", "ValLoss", "GPUMem(MB)"])
    
    for epoch in range(start_epoch, args.epochs):
        l_sum, n = 0.0, 0  # 'l_sum' is epoch sum loss, 'n' is epoch instance number
        model.train()
        scaler = GradScaler()
        #qq=0
        with train_iter as queue:
            batch_fetcher = fi.BatchFetcher(queue)
            for x,y in tqdm.tqdm(batch_fetcher, total=train_iter.iterations_per_epoch,desc='Training'):
                optimizer.zero_grad()
                with autocast(device_type='cuda', dtype=torch.float16):
                    y_pred = model(x).squeeze(1)
                    l = loss(y_pred, y)
                scaler.scale(l).backward()
                scaler.step(optimizer)
                scaler.update()
                l_sum += l.item() * y.shape[0]
                #print(f"train_loss {qq}: {l.item()}")
                #qq+=1
                n += y.shape[0]
                #del x, y_pred,y,l
                #torch.cuda.empty_cache() 
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
        es(val_loss, model, optimizer, scheduler, epoch)
        if es.early_stop:
            print("Early stopping")
            break
        train_iter.reboot()
        val_iter.reboot()
@torch.no_grad()
def test(zscore, loss, model, test_iter, args):
    # Load the model weights
    #model.load_state_dict(torch.load("./Weight/STGCN_" + args.dataset + args.fname + ".pt"))
    

    checkpoint_path = "./Weight/STGCN_" + args.dataset + args.fname + ".pt"
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    test_iter.reboot()
    test_MSE = utility.evaluate_model_multi(model, loss, test_iter,args,device,zscore)
    test_iter.reboot()
    test_MAE, test_RMSE, test_WMAPE = utility.evaluate_metric_OSA_multi(model, test_iter, zscore, device)
     # CSV 저장 준비
    output_file = f"./Result/test_results_{args.dataset+args.fname}.csv"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    test_iter.reboot()
    with test_iter as queue:
        batch_fetcher = fi.BatchFetcher(queue)
        # 첫 배치로 헤더 정보 얻기
        for first_batch in batch_fetcher:
            _, ground_truth = first_batch
            break
        
        # CSV 파일 작성
        with open(output_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            
            # 헤더 작성
            header = []
            for ch in range(ground_truth.size(1)):
                header.extend([f"CH{ch} Ground Truth", f"CH{ch} Prediction"])
            writer.writerow(header)
            
            # 데이터 작성
            with test_iter as queue:
                batch_fetcher = fi.BatchFetcher(queue)
                for batch_idx, (inputs, ground_truth) in enumerate(tqdm.tqdm(batch_fetcher, 
                                                                total=min(16, test_iter.iterations_per_epoch))):
                    if batch_idx >= 16:  # max_batches
                        break
                    inputs=torch.tensor(inputs,dtype=torch.float32)
                    ground_truth=torch.tensor(ground_truth,dtype=torch.float32)
                    predictions = model(inputs).squeeze(1)
                    mid_point = args.batch_size // 2
                    indices = [0, mid_point]
                    
                    # CPU로 이동 및 역정규화
                    predictions = predictions.cpu().numpy()
                    ground_truth = ground_truth.cpu().numpy()
                    
                    for i in range(predictions.shape[1]):
                        predictions[:,i,:] = zscore.inverse_transform(predictions[:,i,:])
                        ground_truth[:,i,:] = zscore.inverse_transform(ground_truth[:,i,:])
                    
                    # 선택된 인덱스의 데이터만 저장
                    for idx in indices:
                        gt_slice = ground_truth[idx, :, :]
                        pred_slice = predictions[idx, :, :]
                        
                        for feature_idx in range(gt_slice.shape[1]):
                            row = []
                            for ch in range(gt_slice.shape[0]):
                                row.append(f"{gt_slice[ch, feature_idx]:.6f}")
                                row.append(f"{pred_slice[ch, feature_idx]:.6f}")
                            writer.writerow(row)
    
    print(f'Dataset {args.dataset:s} | Test loss {test_MSE:.6f} | MAE {test_MAE:.6f} | RMSE {test_RMSE:.6f} | WMAPE {test_WMAPE:.8f}')
    print(f"Test results saved to {output_file}")
    
    return test_MSE, test_MAE, test_RMSE, test_WMAPE
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    
    args, device, blocks = get_parameters()
    
    vel_path = f"./data/{args.dataset}/speed_matrix.h5"
    weight_path = f"./data/{args.dataset}/weight_matrix.h5"
    
    data, args, n_vertex,zscore,train_iter,val_iter,test_iter =setup_preprocess(args,vel_path,weight_path)
    loss, es, model, optimizer, scheduler,start_epoch, best_val_loss = setup_model(args, blocks, n_vertex)
    x,y=data.read_chunk_training_batch(0)
    print(x.shape,y.shape)
    if args.mode == 'train':
        train(args, model, loss, optimizer, scheduler, es, train_iter, val_iter,start_epoch, best_val_loss)
    val_iter.file_manager.__del__()
    train_iter.file_manager.__del__()
    test(zscore, loss, model, test_iter, args)
    test_iter.file_manager.__del__()


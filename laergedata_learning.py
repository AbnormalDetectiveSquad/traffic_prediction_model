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
import file_input as fi
from script import dataloader, utility, earlystopping, opt
from model import models
from torch.amp import autocast, GradScaler
import traceback
import h5py
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
  

    parser.add_argument('--fname', type=str, default='K460_16base_S250samp_seq_lr0.0001', help='name')
    parser.add_argument('--mode', type=str, default='train', help='test or train')
    parser.add_argument('--HotEncoding', type=str, default="On", help='On or Off')
    
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


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    args, device, blocks = get_parameters()

# .h5 파일 경로 지정
    file_path1 = f"./data/{args.dataset}/speed_matrix.h5"
    file_path2 = f"./data/{args.dataset}/weight_matrix.h5"

    # 파일 열기
    data = fi.FileManager(file_path1,file_path2,args)
    zscore = fi.DataNormalizer()
    vel = data.read_chunk(0, 5000, 'vel')
    out=zscore.initialize_from_data(vel)
    data.zscore = zscore
    
    x, sol = data.read_chunk_training_batch(0)

    print(x[0,0,0,0],sol[0,0,0])
    data.__del__()


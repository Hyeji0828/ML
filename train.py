import sys
import os.path
import random
import time
from argparse import ArgumentParser

import torch
import numpy as np
import wandb

def parse_args():
    parser = ArgumentParser()

    parser.add_argument('--seed', type=int, default=28, help='random seed (default=28)')
    parser.add_argument('--epochs', type=int, default=80, help='number of epoch (default=80)')
    parser.add_argument('--model', type=str, default='VGG16', help='model type (default=VGG16)')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size (default=64)')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.2, help='learning rate (default=0.2)')
    parser.add_argument('--optimizer', type=str, default='SGD', help='optimizer type (default=SGD)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--save_interval', type=int, default=10)

    parser.add_argument('--data_dir', type=str, default=os.path.join(os.getcwd(), '/dataset'), help='dataset directory (default=/dataset)')
    parser.add_argument('--model_dir', type=str, default=None, help='directory to save model (default=/name/model)')
    parser.add_argument('--name', type=str, default=None, help='name as identifier (default="model_optimizer_epoch_lr_time")')
    
    args = parser.parse_args()
    if args.name is None:
        cur_time = time.strftime('%y%m%d%H%M')
        args.name = '_'.join([args.model,args.optimizer, str(args.epochs)+'e', 'lr'+str(args.lr), cur_time])
        print(args.name)
    if args.model_dir is None:
        args.model_dir = os.path.join(os.getcwd(),args.name)
    
    return args

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)   # in case using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  
    np.random.seed(seed)
    random.seed(seed)

def train(args):
    # --- load dataset
    # dataset = 

    # --- split data
    # train_loader = 
    # val_loader = 
    
    # --- load model
    device = torch.device(args.device)
    # model = 

    

#       
#       load model, optimizer, scheduler
# 
#       wandb setting
# 
#       train
#           for epoch
#                train step
#
#                cal loss
#           1 epoch done
#           validation
#           save model
#
# }
    pass

def main(args):
    set_seed(args.seed)
    train(args)

if __name__ == "__main__":
    args = parse_args()
    main(args)

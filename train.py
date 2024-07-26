import sys
import os.path
import random
import time
from argparse import ArgumentParser
from importlib import import_module

import torch
from torch.utils.data import DataLoader
import numpy as np
import wandb

from dataset import CustomDataset, CustomTransform

def parse_args():
    parser = ArgumentParser()

    parser.add_argument('--seed', type=int, default=28, help='random seed (default=28)')
    parser.add_argument('--epochs', type=int, default=80, help='number of epoch (default=80)')
    parser.add_argument('--model', type=str, default='VGG16', help='model type (default=VGG16)')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size (default=64)')
    parser.add_argument('--num_workers', type=int, default=8, help='number of workers (default=8)')
    parser.add_argument('--lr', type=float, default=0.2, help='learning rate (default=0.2)')
    parser.add_argument('--optimizer', type=str, default='SGD', help='optimizer type (default=SGD)')
    parser.add_argument('--scheduler', type=str, default='StepLR', help='scheduler type (default=StepLR)')
    parser.add_argument('--save_interval', type=int, default=10, help='model saving interval during training (default=10)')
    parser.add_argument('--resume', type=str, default='auto', help='resume previous training or not (default=auto)')

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

def wandb_setup(args):
    config = {
        'epochs' : args.epochs,
        'architecture' : args.model,
        'batch_size' : args.batch_size,
        'learning_rate' : args.lr,
        'optimizer' : args.optimizer,
        'scheduler' : args.scheduler
    }
    init_kwargs = {
        'project': '',
        'tags' : [],
        'entity' : '',
        'name' : args.name,
        'resume' : args.resume,
        'config' : config
    }
    
    wandb.init(init_kwargs)

def train(args):
    # --- settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # --- load dataset
    tf = CustomTransform()
    train_dataset, val_dataset = CustomDataset(args.data_dir, tf, tf)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=use_cuda,
        drop_last=True)

    val_dataset_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=True)
    
    # --- load model
    model_module = getattr(import_module("model"), args.model)
    model = model_module()

    # --- optimizer
    opt_module = getattr(import_module("optimizer"), args.opimizer)
    optimizer = opt_module(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr = args.lr,
        weight_decay=5e-4
    )
    
    # --- scheduler
    schd_module = getattr(import_module("scheduler"), args.scheduler)
    scheduler = schd_module(optimizer)
    
    # --- wandb setup
    wandb_setup()

    # --- train
    # for epoch in range(args.epochs):
    #     epoch_loss, epoch_start_time = 0, time.time()
    #     model.train()
    
#                train step
#
#                cal loss
#           1 epoch done
#           validation
#           save model
#
# }

def main(args):
    set_seed(args.seed)
    train(args)

if __name__ == "__main__":
    args = parse_args()
    main(args)

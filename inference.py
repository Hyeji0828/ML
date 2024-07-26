import sys
import os.path
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()

    parser.add_argument('--model', type=str, default=None, help='model to load (default=None)')
    parser.add_argument('--option', type=str, default=None, help='ensemble or voting (default=None)')
    parser.add_argument('--models_dir', type=str, default=None, help='dir includes models in case using ensmble or voting (default=None)')

    args = parser.parse_args()
    
    return args

def main(args):
    pass

if __name__ == '__main__':
    args = parse_args()
    main(args)
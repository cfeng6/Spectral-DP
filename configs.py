import torch
import argparse

def set_arguments():
    parser = argparse.ArgumentParser("Spectral-DP")
    parser.add_argument('--root', type=str, default='dataset', help='root dir of all datasets')
    parser.add_argument('--dataset', type=str, default='mnist', help='experiment on dataset in [MNIST]')
    parser.add_argument('--model', type=str, default='mlp_mnist', help='models in [mlp_mnist, lenet_mnist]')
    parser.add_argument('--train_batch_size', type=int, default=250, help='train batch size')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='init learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--gpu', type=bool, default=False, help="if GPU")
    
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--epochs', type=int, default=30)

    parser.add_argument('--block_size', type=int, default=8, help='block size of circulant matrix')
    parser.add_argument('--filter_ratio', type=float, default=0.5, help='Spectral-DP filtering ratio')
    parser.add_argument('--max_grad_norm', type=float, default=0.35, help='gradient clipping')
    parser.add_argument('--epsilon', type=float, default=2)
    parser.add_argument('--delta', type=float, default=1e-5)
    
    args = parser.parse_args()
    if torch.cuda.is_available():
        args.gpu = True

    return args

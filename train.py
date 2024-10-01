import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torchvision.datasets import mnist, CIFAR10

from torchvision.datasets import MNIST

from torch.autograd import Variable
from torch.autograd import gradcheck
import time
import json
import torch.nn.functional as TF
import torch.optim as optim
import os
import math
import matplotlib.pyplot as plt
import pickle
from opacus.utils.uniform_sampler import UniformWithReplacementSampler


from torchsummary import summary
from torch.utils.data import Sampler
from layer_config import DEVICE, SIGMA, FILTER_RATIO, LAYER_BOUND, TOTAL_BOUND

from models.mlp_mnist import MLP
import privacy_budget

from configs import set_arguments
import argparse

# parser = argparse.ArgumentParser("Spectral-DP")
# parser.add_argument('--root', type=str, default='dataset', help='root dir of all datasets')
# parser.add_argument('--dataset', type=str, default='mnist', help='experiment on dataset in [MNIST]')
# parser.add_argument('--model', type=str, default='mlp_mnist')
# parser.add_argument('--train_batch_size', type=int, default=250, help='train batch size')
# parser.add_argument('--learning_rate', type=float, default=0.01, help='init learning rate')
# parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
# parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
# parser.add_argument('--gpu', type=str, default='0', help='gpu device id')

# parser.add_argument('--seed', type=int, default=42, help='random seed')
# parser.add_argument('--epochs', type=int, default=30)

# parser.add_argument('--block_size', type=int, default=8, help='block size of circulant matrix')
# parser.add_argument('--filter_ratio', type=float, default=0.5, help='Spectral-DP filtering ratio')
# parser.add_argument('--max_grad_norm', type=float, default=0.35, help='gradient clipping')
# parser.add_argument('--epsilon', type=float, default=2)
# parser.add_argument('--delta', type=float, default=1e-5)


# args = parser.parse_args()

args = set_arguments()


SAVE_DIR = 'Logs/'

if not os.path.exists(args.root):
    os.mkdir(args.root)

if not os.path.exists(SAVE_DIR):
    os.mkdir(SAVE_DIR)

sub_root = os.path.join(args.root)

if args.dataset == 'mnist':
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        ])
    train_dataset = MNIST(os.path.join(sub_root,'train'), train=True,
                              download=True,transform=transform)
    test_dataset = MNIST(os.path.join(sub_root,'test'), train=False,
                             download=True, transform=test_transform)
else:
     raise Exception("Not a training dataset")
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.train_batch_size)
sample_rate = args.train_batch_size/len(train_dataset)
train_loader = DataLoader(train_dataset,
    batch_sampler=UniformWithReplacementSampler(
        num_samples=len(train_dataset),
        sample_rate=sample_rate,
    ),
)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=100)



if args.model == 'mlp_mnist':
    model = MLP(args.block_size).double().to(DEVICE)
else:
    raise Exception("Not a training model")

optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum,nesterov=True)

cost = nn.CrossEntropyLoss()

train_loss = []
test_loss = []
test_acc = []
SEED = 42
torch.cuda.manual_seed(SEED)
for _epoch in range(args.epochs):
    for idx, (train_x, train_label) in enumerate(train_loader):
        train_x, train_label = train_x.double().to(DEVICE), train_label.to(DEVICE)
        start = time.time()        
        optimizer.zero_grad() 
        outputs = model(train_x)
        loss = cost(outputs, train_label)
        loss.backward()
        optimizer.step()
        if idx % 50 == 0:
            print('idx: {}, loss: {}'.format(idx, loss.sum().item()))
            end = time.time()
            print("epoch:%d, idx:%d, time spent:%.4f s"%(_epoch,idx,(end-start)))
        train_loss.append(loss.sum().item())
    correct = 0
    _sum = 0
    

    for idx, (test_x, test_label) in enumerate(test_loader):
        test_x, test_label = test_x.double().to(DEVICE), test_label.to(DEVICE)
        outputs = model(test_x).detach()
        t_loss = cost(outputs, test_label)
        predict_ys = torch.argmax(outputs, axis=-1)
        _ = predict_ys.detach().data == test_label
        correct += torch.sum(_, axis=-1)
        _sum += _.shape[0]
        test_loss.append(t_loss.sum().item())
        test_acc.append(100*correct / _sum)
    print('Test accuracy: {:.4f}%%'.format(100*correct / _sum))


ckpt = {'net':model.state_dict(),
        'optim':optimizer.state_dict(),
        'sigma':SIGMA}

save_path = os.path.join(SAVE_DIR, 
                         f"{args.dataset}_{args.model}_{args.block_size}_{args.epochs}_{args.epsilon}_{args.filter_ratio}_{args.max_grad_norm}")
log_save_path = os.path.join(SAVE_DIR, 
                         f"{args.dataset}_{args.model}_{args.block_size}_{args.epochs}_{args.epsilon}_{args.filter_ratio}_{args.max_grad_norm}_log")
args_save_path = os.path.join(SAVE_DIR, 
                         f"{args.dataset}_{args.model}_{args.block_size}_{args.epochs}_{args.epsilon}_{args.filter_ratio}_{args.max_grad_norm}_args.txt")



torch.save(ckpt,save_path)
pickle.dump([train_loss, test_loss,test_acc], open(log_save_path,"wb"))
with open(args_save_path, "w") as f:
    json.dump(args.__dict__, f, indent=2)

def main():
    return None

if __name__ == '__main__':
    main()


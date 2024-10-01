import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import mnist, CIFAR10

import torch.optim as optim
import os

from models.simple_cnn import SimpleCNN
from opacus import PrivacyEngine


DATASET_PATH = 'dataset'


def calculate_sigma(dataset, sub_path, device, epochs, epsilon, train_batch_size, delta=1e-5, max_grad_norm=1.0):
    if dataset == 'mnist':
        transform = transforms.Compose([
            transforms.Resize((32,32)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
    
    train_dataset = mnist.MNIST(root=os.path.join(sub_path, 'train'), train=True,
                              download=True,transform=transform)
    model = SimpleCNN().double().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size)
    privacy_engine = PrivacyEngine()

    model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
              module=model,
              optimizer=optimizer,
              data_loader=train_loader,
              epochs=epochs,
              target_epsilon=epsilon,
              target_delta=delta,
              max_grad_norm=max_grad_norm,
            )
    print(f"Using sigma={optimizer.noise_multiplier} and C={max_grad_norm}")

    return optimizer.noise_multiplier
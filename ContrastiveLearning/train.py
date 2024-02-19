import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm

from ntxent_loss import NTXentLoss
from resnet_simclr import ResNetSimCLR

class SimCLRTransform:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(size=28),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1)
            ], p=0.8),
            #GaussianBlur(kernel_size=int(0.1 * 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def __call__(self, x):
        return self.transform(x), self.transform(x)

def train(model, train_loader, optimizer, loss_fn, epochs, device):
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for (x_i, x_j), _ in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}', unit='batch'):
            x_i, x_j = x_i.to(device), x_j.to(device)

            # Forward pass
            z_i, z_j = model(x_i, x_j)
            loss = loss_fn(z_i, z_j)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{epochs}]: Average Loss: {avg_loss:.4f}')

if __name__ == '__main__':
    # Set random seeds and device
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    train_dataset = datasets.FashionMNIST('./data', train=True, download=True, transform=SimCLRTransform())
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    model = ResNetSimCLR(out_dim=128).to(device)

    loss_fn = NTXentLoss(batch_size=128, temperature=0.5, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    # Train model
    train(model, train_loader, optimizer, loss_fn, epochs=2, device=device)

    # Save model
    # save_checkpoint({
    #     'epoch': 10,
    #     'state_dict': model.state_dict(),
    #     'optimizer': optimizer.state_dict(),
    # }, is_best=True)

import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.utils.data.dataset import Subset

from tqdm import tqdm
import matplotlib.pyplot as plt
from ntxent_loss import NTXentLoss
from resnet_simclr import ResNetSimCLR
from utils import create_subset

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SimCLRTransform:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(size=28), # recommend to stand out in SimCLR paper 
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1) # recommend to stand out in SimCLR paper 
            ], p=0.8),
            #GaussianBlur(kernel_size=int(0.1 * 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def __call__(self, x):
        return self.transform(x), self.transform(x)


class SimCLREncoderClassifier(nn.Module):
    def __init__(self, simclr_model, num_classes=10):
        super(SimCLREncoderClassifier, self).__init__()
        self.simclr_encoder = simclr_model.resnet 
        
        for param in self.simclr_encoder.parameters():
            param.requires_grad = False

        # Assuming the feature dimension from the encoder
        feature_dim = 512

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        features = self.simclr_encoder(x)
        features = torch.flatten(features, 1)
        return self.classifier(features)

    
class SimCLRTrainer:
    def __init__(self, model, train_loader, optimizer, loss_fn, epochs):
        self.model = model
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.epochs = epochs
        self.device = device

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        for (x_i, x_j), _ in tqdm(self.train_loader, desc="Training SimCLR", unit="batch"):
            x_i, x_j = x_i.to(self.device), x_j.to(self.device)
            z_i, z_j = self.model(x_i, x_j)
            
            loss = self.loss_fn(z_i, z_j)
            self.optimizer.zero_grad()
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        avg_loss = total_loss / len(self.train_loader)
        print(f'Average Loss: {avg_loss:.4f}')

    def train(self):
        for epoch in range(self.epochs):
            print(f'Epoch {epoch+1}/{self.epochs}')
            self.train_epoch()

class ClassifierTrainer:
    def __init__(self, model, train_loader, val_loader, epochs):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.device = device
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=1e-3)
        self.loss_fn = nn.CrossEntropyLoss()
        self.train_losses = []
        self.val_losses = []
        self.train_accuracy = []
        self.val_accuracy = []

    def train_epoch(self):
        self.model.train()
        total_loss, total, correct = 0, 0, 0
        for data, labels in tqdm(self.train_loader, desc="Training"):
            data, labels = data.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(data)
            loss = self.loss_fn(outputs, labels)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100 * correct / total
        self.train_losses.append(avg_loss)
        self.train_accuracy.append(accuracy)
        print(f'Train Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')

    def validate(self):
        self.model.eval()
        total_loss, total, correct = 0, 0, 0
        with torch.no_grad():
            for data, labels in tqdm(self.val_loader, desc="Validating"):
                data, labels = data.to(self.device), labels.to(self.device)
                outputs = self.model(data)
                loss = self.loss_fn(outputs, labels)
                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100 * correct / total
        self.val_losses.append(avg_loss)
        self.val_accuracy.append(accuracy)
        print(f'Validation Loss: {avg_loss:.4f}, Validation Accuracy: {accuracy:.2f}%')

    def train(self):
        for epoch in range(self.epochs):
            print(f'Epoch {epoch + 1}/{self.epochs}')
            self.train_epoch()
            self.validate()
        self.plot_metrics()

    def plot_metrics(self,filename='../assets/SimCLR_training_validation_metrics.png'):
        epochs_range = range(1, self.epochs + 1)
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, self.train_losses, label='Train Loss')
        plt.plot(epochs_range, self.val_losses, label='Validation Loss')
        plt.title('Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, self.train_accuracy, label='Train Accuracy')
        plt.plot(epochs_range, self.val_accuracy, label='Validation Accuracy')
        plt.title('Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()

        plt.tight_layout()
        plt.savefig(filename)
        plt.show()

if __name__ == "__main__":
    # SimCLR model training setup
    train_dataset = datasets.FashionMNIST('./data', train=True, download=True, transform=SimCLRTransform())
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    model_simclr = ResNetSimCLR(out_dim=512).to(device) 
    optimizer_simclr = optim.Adam(model_simclr.parameters(), lr=3e-4)
    loss_fn_simclr = NTXentLoss(temperature=0.5, device=device)

    simclr_trainer = SimCLRTrainer(model_simclr, train_loader, optimizer_simclr, loss_fn_simclr, epochs=10)
    simclr_trainer.train()


    # Classifier training setup
    training_set = datasets.FashionMNIST('./data', train=True, download=True, transform=transforms.ToTensor())
    validation_set = datasets.FashionMNIST('./data', train=False, download=True, transform=transforms.ToTensor())

    subset_training_set = create_subset(training_set, subset_size=0.1)
    
    train_loader_subset = DataLoader(subset_training_set, batch_size=64, shuffle=True)
    validation_loader = DataLoader(validation_set, batch_size=64, shuffle=False)

    simclr_classifier = SimCLREncoderClassifier(model_simclr, num_classes=10).to(device)  # Assume this is defined correctly

    classifier_trainer = ClassifierTrainer(simclr_classifier, train_loader_subset, validation_loader, epochs=10)
    classifier_trainer.train()
    classifier_trainer.plot_metrics()

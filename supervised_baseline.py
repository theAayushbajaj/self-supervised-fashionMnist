import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import seaborn as sns
from barbar import Bar
import json
import os

if not os.path.exists('assets'):
    os.makedirs('assets')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class FashionMNISTClassifier(nn.Module):
    def __init__(self):
        super(FashionMNISTClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Trainer:
    def __init__(self, train_loader, validation_loader, epochs):
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.epochs = epochs
        self.model = FashionMNISTClassifier().to(device)
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        self.train_losses = []
        self.validation_losses = []
        self.train_accuracy = []
        self.validation_accuracy = []

    def train_one_epoch(self):
        self.model.train()  # Set model to training mode
        running_loss = 0.0
        correct = 0
        total = 0
        for i, data in enumerate(Bar(self.train_loader)):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        avg_loss = running_loss / len(self.train_loader)
        accuracy = 100 * correct / total
        self.train_losses.append(avg_loss)
        self.train_accuracy.append(accuracy)
        print(f'Train Loss: {avg_loss:.4f}, Train Accuracy: {accuracy:.2f}%')

    def validate(self):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_predicted = []
        all_labels = []
        with torch.no_grad():
            for data in Bar(self.validation_loader):
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = self.model(images)
                loss = self.loss_fn(outputs, labels)
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_predicted.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_loss = running_loss / len(self.validation_loader)
        accuracy = 100 * correct / total
        self.validation_losses.append(avg_loss)
        self.validation_accuracy.append(accuracy)
        print(f'Validation Loss: {avg_loss:.4f}, Validation Accuracy: {accuracy:.2f}%')
        
        # Save all_labels and all_predicted to use in the confusion matrix
        self.all_labels = all_labels
        self.all_predicted = all_predicted

    def train(self):
        for epoch in range(self.epochs):
            print(f'Starting Epoch {epoch+1}/{self.epochs}')
            self.train_one_epoch()
            self.validate()

    def plot_metrics(self, filename='assets/train_val_metrics.png'):
        epochs = range(1, self.epochs + 1)
        plt.figure(figsize=(12, 6))

        # Plot Training and Validation Loss
        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.train_losses, 'r-', label='Training Loss')
        plt.plot(epochs, self.validation_losses, 'b-', label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        # Plot Training and Validation Accuracy
        plt.subplot(1, 2, 2)
        plt.plot(epochs, self.train_accuracy, 'r-', label='Training Accuracy')
        plt.plot(epochs, self.validation_accuracy, 'b-', label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.tight_layout()
        plt.savefig(filename)
        plt.close()


    def plot_confusion_matrix(self, filename='assets/baseline_confusion_matrix.png'):
        cm = confusion_matrix(self.all_labels, self.all_predicted)
        plt.figure(figsize=(10, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.show()
        plt.savefig(filename)


if __name__ == '__main__':
    # Data preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Datasets and Dataloaders
    training_set = torchvision.datasets.FashionMNIST('./data', train=True, download=True, transform=transform)
    validation_set = torchvision.datasets.FashionMNIST('./data', train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(training_set, batch_size=64, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=64, shuffle=False)

    # Initialize our trainer and start training
    trainer = Trainer(train_loader=train_loader, 
                      validation_loader=validation_loader, 
                      epochs=20)
    trainer.train()  # Train for 5 epochs
    trainer.plot_metrics('baseline_metrics.png')  # Plot and save the training and validation loss and accuracy
    trainer.plot_confusion_matrix()  # Plot and save the confusion matrix

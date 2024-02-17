import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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
    def __init__(self, train_loader, epochs):
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.epochs = epochs
        self.model = FashionMNISTClassifier()
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        self.train_losses = []
        self.validation_losses = []
        self.train_accuracy = []
        self.validation_accuracy = []

    def train_one_epoch(self):
        running_loss = 0.0
        correct = 0
        total = 0
        for i, data in enumerate(self.train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to('device'), labels.to('device')
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

    def validate(self):
        running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for data in self.validation_loader:
                images, labels = data
                outputs = self.model(images)
                loss = self.loss_fn(outputs, labels)
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_loss = running_loss / len(self.validation_loader)
        accuracy = 100 * correct / total
        self.validation_losses.append(avg_loss)
        self.validation_accuracy.append(accuracy)

    def train(self):
        for epoch in range(self.epochs):
            self.model.train()
            self.train_one_epoch()
            self.model.eval()
            self.validate()
            print(f'Epoch {epoch+1}, Train Loss: {self.train_losses[-1]}, Train Accuracy: {self.train_accuracy[-1]}, Validation Loss: {self.validation_losses[-1]}, Validation Accuracy: {self.validation_accuracy[-1]}')

    def plot_metrics(self, filename='training_validation_metrics.png'):
        plt.figure(figsize=(12, 6))
        
        # Plot Training and Validation Loss
        plt.subplot(1, 2, 1)
        plt.plot(self.epochs, self.train_losses, 'r-', label='Training Loss')
        plt.plot(self.epochs, self.validation_losses, 'b-', label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        # Plot Training and Validation Accuracy
        plt.subplot(1, 2, 2)
        plt.plot(self.epochs, self.train_accuracy, 'r-', label='Training Accuracy')
        plt.plot(self.epochs, self.validation_accuracy, 'b-', label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        # Save the figure
        plt.tight_layout()  # Adjust subplots to fit into figure area.
        plt.savefig(filename)
        plt.close()  # Close the figure to free up memory


if __name__ == '__main__':
    # Data preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Datasets and Dataloaders
    training_set = torchvision.datasets.FashionMNIST('./data', train=True, download=True, transform=transform)
    validation_set = torchvision.datasets.FashionMNIST('./data', train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(training_set, batch_size=4, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=4, shuffle=False)

    # Initialize our trainer and start training
    trainer = FashionMNISTClassifier(train_loader = train_loader, 
                                     validation_loader = validation_loader,
                                     epochs = 5)
    trainer.train()  # Train for 5 epochs
    trainer.plot_metrics()  # Plot the training and validation loss and accuracy

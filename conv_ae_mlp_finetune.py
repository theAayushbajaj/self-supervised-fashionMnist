import torch
from torch import nn, optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import logging
from utils import create_subset
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
    
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),  # [batch, 16, 14, 14]
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # [batch, 32, 7, 7]
            nn.ReLU(True),
            nn.Conv2d(32, 64, 7)  # [batch, 64, 1, 1]
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 7),  # [batch, 32, 7, 7]
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),  # [batch, 16, 14, 14]
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),  # [batch, 1, 28, 28]
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class ConvEncoderClassifier(nn.Module):
    def __init__(self, encoder, num_classes=10):
        super(ConvEncoderClassifier, self).__init__()
        self.encoder = encoder
        for param in self.encoder.parameters():
            param.requires_grad = False  # Freeze the encoder
        
        # Assuming encoder output is 64-dimensional
        self.classifier = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.encoder(x)  # Encoded representations
        x = torch.flatten(x, 1)  # Flatten the output for the classifier
        x = self.classifier(x)
        return x


class AutoencoderTrainer:
    def __init__(self, model, train_loader, val_loader, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.loss_fn = nn.MSELoss()
        self.train_losses = []
        self.val_losses = []

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        for data, _ in self.train_loader:
            data = data.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.loss_fn(output, data)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(self.train_loader)
        self.train_losses.append(avg_loss)
        logging.info(f'Training Loss: {avg_loss:.4f}')

    def validate(self):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for data, _ in self.val_loader:
                data = data.to(self.device)
                output = self.model(data)
                loss = self.loss_fn(output, data)
                total_loss += loss.item()
        avg_loss = total_loss / len(self.val_loader)
        self.val_losses.append(avg_loss)
        logging.info(f'Validation Loss: {avg_loss:.4f}')

    def train(self, epochs):
        for epoch in range(epochs):
            logging.info(f'Epoch {epoch + 1}/{epochs}')
            self.train_epoch()
            self.validate()

    # def plot_metrics(self):
    #     plt.figure(figsize=(10, 5))
    #     plt.plot(self.train_losses, label='Train Loss')
    #     plt.plot(self.val_losses, label='Validation Loss')
    #     plt.title('Loss over epochs')
    #     plt.xlabel('Epochs')
    #     plt.ylabel('Loss')
    #     plt.legend()
    #     plt.show()


class ClassifierTrainer:
    def __init__(self, model, train_loader, val_loader, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=1e-3)
        self.loss_fn = nn.CrossEntropyLoss()
        self.train_losses = []
        self.val_losses = []
        self.train_accuracy = []
        self.val_accuracy = []

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        for data, labels in self.train_loader:
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
        logging.info(f'Train Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')

    def validate(self):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for data, labels in self.val_loader:
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
        logging.info(f'Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')

    def train(self, epochs):
        for epoch in range(epochs):
            logging.info(f'Epoch {epoch + 1}/{epochs} - Training')
            self.train_epoch()
            logging.info(f'Epoch {epoch + 1}/{epochs} - Validation')
            self.validate()

    def plot_metrics(self, filename='conv_ae_mlp_metrics.png'):
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.title('Training & Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(self.train_accuracy, label='Train Accuracy')
        plt.plot(self.val_accuracy, label='Validation Accuracy')
        plt.title('Training & Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.tight_layout()

        plt.savefig(filename)
        plt.close()


if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Datasets and Dataloaders
    training_set = torchvision.datasets.FashionMNIST('./data', train=True, download=True, transform=transform)
    validation_set = torchvision.datasets.FashionMNIST('./data', train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(training_set, batch_size=64, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=64, shuffle=False)

    # Train the autoencoder
    autoencoder = ConvAutoencoder().to(device)
    autoencoder_trainer = AutoencoderTrainer(autoencoder, train_loader, device)
    autoencoder_trainer.train(epochs=10)

    # Assuming the classifier is set up properly to use the encoded features
    # Train the classifier
    subset_training_set = create_subset(training_set, subset_size=0.1)

    # Now, use this subset for training
    train_loader = torch.utils.data.DataLoader(subset_training_set, batch_size=64, shuffle=True)

    classifier = ConvEncoderClassifier(autoencoder.encoder, num_classes=10)
    classifier_trainer = ClassifierTrainer(classifier, train_loader, device, epochs=10)
    classifier_trainer.train()
    classifier_trainer.plot_metrics()
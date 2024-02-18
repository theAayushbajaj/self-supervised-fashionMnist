import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn, optim
from torch.utils.data import DataLoader
import logging
from resnet18_vae import VAEClassifier, VAE 
from utils import create_subset
import matplotlib.pyplot as plt

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class VAETrainer:
    def __init__(self, train_loader, val_loader, lr=1e-3, z_dim=10, device=None):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = VAE(z_dim=z_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss(reduction='sum')

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        for batch_idx, (data, _) in enumerate(self.train_loader):
            data = data.to(self.device)
            self.optimizer.zero_grad()
            reconstruction, mu, logvar = self.model(data)
            loss = self.loss_function(reconstruction, data, mu, logvar)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(self.train_loader.dataset)
        logging.info(f'Train Epoch: Average Loss: {avg_loss:.4f}')

    def validate(self):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for data, _ in self.val_loader:
                data = data.to(self.device)
                reconstruction, mu, logvar = self.model(data)
                loss = self.loss_function(reconstruction, data, mu, logvar)
                total_loss += loss.item()
        avg_loss = total_loss / len(self.val_loader.dataset)
        logging.info(f'Validation: Average Loss: {avg_loss:.4f}')
        return avg_loss

    def train(self, epochs):
        for epoch in range(epochs):
            logging.info(f'Epoch {epoch+1}/{epochs}')
            self.train_epoch()
            val_loss = self.validate()

    def loss_function(self, recon_x, x, mu, logvar):
        BCE = self.loss_fn(recon_x, x)  # Reconstruction loss
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())  # KL divergence loss
        return BCE + KLD

class ClassifierTrainer:
    def __init__(self, train_loader, val_loader, encoder, z_dim, hidden_dim, output_dim, lr=1e-3, device=None):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.encoder = encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        self.model = VAEClassifier(self.encoder, z_dim, hidden_dim, output_dim).to(self.device)
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr)
        
        self.loss_fn = nn.CrossEntropyLoss()

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
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        avg_loss = total_loss / len(self.train_loader.dataset)
        accuracy = 100. * correct / total
        logging.info(f'Train Epoch: Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')

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
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        avg_loss = total_loss / len(self.val_loader.dataset)
        accuracy = 100. * correct / total
        logging.info(f'Validation: Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')

    def train(self, epochs):
        for epoch in range(epochs):
            logging.info(f'Epoch {epoch+1}/{epochs} - Training')
            self.train_epoch()
            logging.info(f'Epoch {epoch+1}/{epochs} - Validation')
            self.validate()

    def plot_metrics(self, filename='training_validation_metrics.png'):
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

    trainer = VAETrainer(train_loader, validation_loader, lr=1e-3, z_dim=10)
    trainer.train(epochs=2)


    # Assuming you've already loaded the full FashionMNIST training dataset as `training_set`
    subset_training_set = create_subset(training_set, subset_size=0.1)

    # Now, use this subset for training
    train_loader = torch.utils.data.DataLoader(subset_training_set, batch_size=64, shuffle=True)

    # train classifier while freezing the encoder
    encoder = trainer.model.encoder
    classifier_trainer = ClassifierTrainer(train_loader, validation_loader, encoder, z_dim=10, hidden_dim=100, output_dim=10)
    classifier_trainer.train(epochs=2)
    classifier_trainer.plot_metrics('vae_mlp_classifier_metrics.png')

    

import torch
import torch.nn as nn
import torch.nn.functional as F

class NTXentLoss(nn.Module):
    def __init__(self, batch_size, temperature, device):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    # def forward(self, z_i, z_j):
    #     N = 2 * self.batch_size  # Corrected size for 2N
    #     z = torch.cat((z_i, z_j), dim=0)

    #     z_norm = F.normalize(z, p=2, dim=1)
    #     sim_matrix = torch.mm(z_norm, z_norm.transpose(0, 1)) / self.temperature

    #     # Create an identity mask to zero-out diagonals
    #     mask = torch.eye(N, device=self.device)
    #     sim_matrix = sim_matrix.masked_fill(mask == 1, -1e9)

    #     # Labels for cross-entropy: each sample should match to its pair
    #     labels = torch.cat([torch.arange(self.batch_size) for _ in range(2)], dim=0)
    #     labels = (labels + self.batch_size) % (2 * self.batch_size)
    #     labels = labels.to(self.device)

    #     # Compute cross-entropy loss
    #     loss = self.criterion(sim_matrix, labels)
    #     return loss / (2 * self.batch_size)
    
    def forward(self, features):

        labels = torch.cat([torch.arange(self.batch_size) for _ in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.device)

        z_norm = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(z_norm, z_norm.T)

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)

        logits = logits / self.temperature
        loss = self.criterion(logits, labels)
        return loss



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

    def forward(self, z_i, z_j):
        print(z_i.shape, z_j.shape)
        N = 2 * self.batch_size  # Corrected size for 2N
        z = torch.cat((z_i, z_j), dim=0)

        z_norm = F.normalize(z, dim=1)
        sim_matrix = torch.mm(z_norm, z_norm.T) / self.temperature

        # Create an identity mask to zero-out diagonals
        mask = torch.eye(N, device=self.device)
        sim_matrix = sim_matrix.masked_fill(mask == 1, -1e9)

        # Labels for cross-entropy: each sample should match to its pair
        labels = torch.cat([torch.arange(self.batch_size) for _ in range(2)], dim=0)
        labels = (labels + self.batch_size) % (2 * self.batch_size)
        labels = labels.to(self.device)

        # Compute cross-entropy loss
        loss = self.criterion(sim_matrix, labels)
        return loss / (2 * self.batch_size)
    



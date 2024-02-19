import torch
import torch.nn as nn
import torch.nn.functional as F

class NTXentLoss(nn.Module):
    def __init__(self, temperature, device):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.device = device
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def forward(self, z_i, z_j):
        N = z_i.size(0) 
        z = torch.cat((z_i, z_j), dim=0)  # Concatenated embeddings

        z_norm = F.normalize(z, dim=1)
        sim_matrix = torch.mm(z_norm, z_norm.T) / self.temperature

        # Mask out self-similarities by setting diagonal to a large negative value
        sim_matrix.fill_diagonal_(-1e9)

        # Labels: For each image in z_i, its positive pair is its counterpart in z_j, and vice versa
        labels = torch.arange(2*N, device=self.device)
        labels = (labels + N) % (2*N)  # Shift labels to match positive pairs

        logits = sim_matrix
        loss = self.criterion(logits, labels)

        return loss / (2 * N) 
    



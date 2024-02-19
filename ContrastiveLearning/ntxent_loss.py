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
        N = 2 * self.batch_size
        z = torch.cat((z_i, z_j), dim=0)

        sim_matrix = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2) / self.temperature
        sim_matrix = sim_matrix - torch.eye(N, device=self.device) * 1e9

        sim_matrix.fill_diagonal_(-9e15)

        # Create positive pairs, with main diagonal elements being the ground truth (where i=j)
        positives = torch.exp(sim_matrix / self.temperature)
        #positives = torch.exp(torch.sum(z_i * z_j, dim=1) / self.temperature)
        negatives = torch.sum(torch.exp(sim_matrix / self.temperature), dim=1)

        loss_per_sample = -torch.log(positives / negatives)

        loss = torch.sum(loss_per_sample) / (2 * self.batch_size)
        return loss



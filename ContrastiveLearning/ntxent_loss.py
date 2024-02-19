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
    #     z = torch.cat((z_i, z_j), dim=0) 
    #     N = 2 * self.batch_size  # Corrected size for 2N, Size for concatenated z_i and z_j

    #     z_norm = F.normalize(z, dim=1)
    #     sim_matrix = torch.mm(z_norm, z_norm.T) / self.temperature

    #     # Create an identity mask to zero-out diagonals
    #     mask = torch.eye(self.batch_size, device=self.device)
    #     sim_matrix = sim_matrix.masked_fill(mask == 1, -1e9)

    #     labels = torch.cat([torch.arange(self.batch_size) for _ in range(2)], dim=0)
    #     labels = (labels + self.batch_size) % (2 * self.batch_size)
    #     labels = labels.to(self.device)

    #     # Compute cross-entropy loss
    #     loss = self.criterion(sim_matrix, labels)
    #     return loss / (2 * self.batch_size)
    
    def forward(self, z_i, z_j):
        batch_size = z_i.size(0)
        N = 2 * batch_size  # Size for concatenated z_i and z_j
        
        z = torch.cat((z_i, z_j), dim=0)
        z_norm = F.normalize(z, dim=1)
        sim_matrix = torch.mm(z_norm, z_norm.T) / self.temperature
        
        # Ensure diagonal elements are not selected as negatives
        sim_matrix.fill_diagonal_(-1e9)
        
        labels = torch.cat([torch.arange(batch_size) for _ in range(2)], dim=0)
        labels = labels.to(self.device)
        
        # Duplicate labels to match the size of sim_matrix
        labels = torch.cat((labels, labels), dim=0)
        
        sim_matrix = torch.cat((sim_matrix, sim_matrix), dim=1)
        
        loss = self.criterion(sim_matrix, labels)
        return loss / (2 * batch_size)
    



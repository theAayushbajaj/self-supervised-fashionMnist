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
        N = z_i.size(0)  # Original batch size
        z = torch.cat((z_i, z_j), dim=0)  # Concatenated embeddings

        # Compute similarity matrix
        z_norm = F.normalize(z, dim=1)
        sim_matrix = torch.mm(z_norm, z_norm.T) / self.temperature

        # Mask out self-similarities by setting diagonal to a large negative value
        sim_matrix.fill_diagonal_(-1e9)

        # Labels: For each image in z_i, its positive pair is its counterpart in z_j, and vice versa
        labels = torch.arange(2*N, device=self.device)
        labels = (labels + N) % (2*N)  # Shift labels to match positive pairs

        # The logits are the similarity scores; targets are the indices of the positive pairs
        logits = sim_matrix
        loss = self.criterion(logits, labels)

        return loss / (2 * N)  # Normalize by the effective batch size (2N)
    
    # def forward(self, z_i, z_j):
    #     z = torch.cat((z_i, z_j), dim=0) 
    #     N = 2 * self.batch_size  # Corrected size for 2N, Size for concatenated z_i and z_j

    #     z_norm = F.normalize(z, dim=1)
    #     sim_matrix = torch.mm(z_norm, z_norm.T) / self.temperature

    #     # Create an identity mask to zero-out diagonals
    #     sim_matrix.fill_diagonal_(-1e9)

    #     labels = torch.arange(N, device=self.device)
    #     labels = torch.cat((labels, labels), dim=0) % N

    #     #sim_matrix = torch.cat((sim_matrix, sim_matrix), dim=1)
        
    #     loss = self.criterion(sim_matrix, labels)
    #     return loss / (2 * self.batch_size)
    
    # def forward(self, z_i, z_j):
    #     batch_size = z_i.size(0)
    #     N = 2 * batch_size  # Size for concatenated z_i and z_j
        
    #     z = torch.cat((z_i, z_j), dim=0)
    #     z_norm = F.normalize(z, dim=1)
    #     sim_matrix = torch.mm(z_norm, z_norm.T) / self.temperature
        
    #     # Ensure diagonal elements are not selected as negatives
    #     sim_matrix.fill_diagonal_(-1e9)
        
    #     labels = torch.cat([torch.arange(batch_size) for _ in range(2)], dim=0)
    #     labels = labels.to(self.device)
        
    #     # Duplicate labels to match the size of sim_matrix
    #     labels = torch.cat((labels, labels), dim=0)
        
    #     sim_matrix = torch.cat((sim_matrix, sim_matrix), dim=1)
        
    #     loss = self.criterion(sim_matrix, labels)
    #     return loss / (2 * batch_size)
    



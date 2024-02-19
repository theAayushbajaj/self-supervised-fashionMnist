import torch.nn as nn
import torchvision.models as models

class ProjectionHead(nn.Module):
    def __init__(self, in_dim=128, out_dim=128):
        super(ProjectionHead, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(),
            nn.Linear(512, out_dim),
        )
        
    def forward(self, x):
        return self.layers(x)

    
class ResNetSimCLR(nn.Module):
    def __init__(self, out_dim=128):
        super(ResNetSimCLR, self).__init__()
        self.resnet = models.resnet18(weights=None)

        dim_mlp = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()

        self.projection_head = ProjectionHead(in_dim=dim_mlp, out_dim=out_dim)

    def forward(self, x_i, x_j):
        h_i = self.resnet(x_i)
        h_j = self.resnet(x_j)
        z_i = self.projection_head(h_i)
        z_j = self.projection_head(h_j)
        return z_i, z_j
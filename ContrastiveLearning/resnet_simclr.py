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
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        dim_mlp = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()

        self.projection_head = ProjectionHead(in_dim=dim_mlp, out_dim=out_dim)

    def forward(self, x):
        h = self.resnet(x)
        z = self.projection_head(h)
        return z
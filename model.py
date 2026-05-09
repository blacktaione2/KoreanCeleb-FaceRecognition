import torch.nn as nn
from torchvision import models

class MultiTaskResNet(nn.Module):
    def __init__(self, num_celebs, pretrained=True, dropout_rate=0.3, freeze_backbone=False):
        super(MultiTaskResNet, self).__init__()

        # 1. Backbone
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.resnet18(weights=weights)
        
        in_features = self.backbone.fc.in_features # 512
        self.backbone.fc = nn.Identity()

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # 2. Shared Representation Layer (차원 확장: 256 -> 512)
        # 512차원을 유지하여 정보 손실을 최소화, LayerNorm으로 안정성 증가
        self.shared = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.LayerNorm(512), 
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate)
        )

        # 3. Task Specific Heads
        self.celeb_head = nn.Linear(512, num_celebs)
        self.smile_head = nn.Linear(512, 2)

    def forward(self, x):
        features = self.backbone(x)         
        shared_feat = self.shared(features)    
        
        celeb_out = self.celeb_head(shared_feat)
        smile_out = self.smile_head(shared_feat)
        
        return celeb_out, smile_out


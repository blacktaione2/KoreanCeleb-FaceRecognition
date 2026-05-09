import torch.nn as nn
from torchvision import models

class MultiTaskMobileNet(nn.Module):
    def __init__(self, num_celebs, pretrained=True, dropout_rate=0.3, freeze_backbone=False):
        super(MultiTaskMobileNet, self).__init__()

        # 1. Backbone
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.mobilenet_v2(weights=weights)

        in_features = self.backbone.classifier[1].in_features  # 1280
        self.backbone.classifier = nn.Identity()               # classifier 제거, 1280 특징 벡터 출력

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # 2. Shared Representation Layer
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
    
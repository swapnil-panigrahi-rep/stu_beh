import torch.nn as nn
import timm

class EmotionModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.model = timm.create_model(
            "vit_base_patch16_224",
            pretrained=True
        )

        self.model.head = nn.Linear(
            self.model.head.in_features,
            7
        )

    def forward(self,x):

        return self.model(x)

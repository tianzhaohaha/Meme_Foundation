"""TumEmo classifiers with a replaceable head and explicit backbone freezing."""
import os
os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"
import torch
from torch import nn
from torchvision.models import vit_b_16, ViT_B_16_Weights
from transformers import AutoModel
IMAGE_MODEL = 'vit_b_16'
IMAGE_WEIGHTS = 'IMAGENET1K_V1'
TEXT_MODEL = 'FacebookAI/roberta-base'
class EmotionModel(nn.Module):
    def __init__(self, pretrained=True, dropout=0.5, num_classes=2, architecture='multimodal'):
        super().__init__()
        if architecture not in ('multimodal', 'vit'):
            raise ValueError(f'Unknown architecture: {architecture}')
        self.architecture = architecture
        self.backbone_frozen = False
        self.image_encoder = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None)
        self.image_encoder.heads = nn.Identity()
        if architecture == 'vit':
            self.text_encoder = None
        elif pretrained:
            self.text_encoder = AutoModel.from_pretrained(TEXT_MODEL, add_pooling_layer=False)
        else:
            from transformers import RobertaConfig, RobertaModel
            self.text_encoder = RobertaModel(RobertaConfig(), add_pooling_layer=False)
        self.fusion = nn.Sequential(nn.Linear(1536 if architecture == 'multimodal' else 768, 256), nn.LayerNorm(256), nn.ReLU())
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(256, num_classes))

    def freeze_backbone(self):
        """Freeze encoders AND fusion; only the newly initialized head is trained."""
        self.backbone_frozen = True
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(name.startswith('head.'))
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        if self.backbone_frozen:
            self.image_encoder.eval()
            if self.text_encoder is not None:
                self.text_encoder.eval()
            self.fusion.eval()
        return self

    def features(self, images, input_ids, attention_mask):
        image = self.image_encoder(images)
        if self.text_encoder is None:
            return self.fusion(image)
        hidden = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        text = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
        return self.fusion(torch.cat((image, text), dim=1))

    def forward(self, images, input_ids, attention_mask):
        if self.backbone_frozen:
            with torch.no_grad():
                features = self.features(images, input_ids, attention_mask)
        else:
            features = self.features(images, input_ids, attention_mask)
        return self.head(features)

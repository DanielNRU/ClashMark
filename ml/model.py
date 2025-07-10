import torch
from torchvision import models
 
# Модель используется для инференса только для visual-пар (см. алгоритм в routes.py/inference.py)
def create_model(device):
    model = models.mobilenet_v3_small(weights='IMAGENET1K_V1')
    model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, 1)
    model = model.to(device)
    return model 
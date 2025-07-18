import torch
from torchvision import models
 
# Модель используется для инференса только для visual-пар (см. алгоритм в routes.py/inference.py)
def create_model(device):
    model = models.mobilenet_v3_small(weights='IMAGENET1K_V1')
    model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, 1)
    model = model.to(device)
    return model 

_MODEL_CACHE = {}
def get_cached_model(device, model_file):
    key = (str(device), model_file)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    model = create_model(device)
    import torch
    model.load_state_dict(torch.load(model_file, map_location=device))
    model.to(device)
    model.eval()
    _MODEL_CACHE[key] = model
    return model 
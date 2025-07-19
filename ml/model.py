import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

def create_model(device, model_type='mobilenet_v3_small'):
    """
    Создает модель с выбором архитектуры для оптимального баланса скорости и качества.
    
    Args:
        device: устройство для модели
        model_type: тип архитектуры ('mobilenet_v3_small', 'efficientnet_b0', 'resnet18')
    """
    if model_type == 'mobilenet_v3_small':
        # MobileNetV3 Small - оптимален для скорости
        model = models.mobilenet_v3_small(weights='IMAGENET1K_V1')
        # Заменяем последний слой для бинарной классификации
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, 1)
        
    elif model_type == 'efficientnet_b0':
        # EfficientNet-B0 - хороший баланс скорости и качества
        model = models.efficientnet_b0(weights='IMAGENET1K_V1')
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
        
    elif model_type == 'resnet18':
        # ResNet18 - классический выбор для качества
        model = models.resnet18(weights='IMAGENET1K_V1')
        model.fc = nn.Linear(model.fc.in_features, 1)
        
    elif model_type == 'mobilenet_v2':
        # MobileNetV2 - альтернатива V3
        model = models.mobilenet_v2(weights='IMAGENET1K_V1')
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
        
    else:
        raise ValueError(f"Неизвестный тип модели: {model_type}")
    
    model = model.to(device)
    return model

def create_optimized_model(device, model_type='mobilenet_v3_small'):
    """
    Создает оптимизированную модель с дополнительными улучшениями для инференса.
    """
    model = create_model(device, model_type)
    
    # Оптимизации для инференса
    model.eval()  # Переводим в режим оценки
    
    return model

_MODEL_CACHE = {}

def get_cached_model(device, model_file, model_type='mobilenet_v3_small', use_optimization=True):
    """
    Получает кэшированную модель с возможностью оптимизации.
    
    Args:
        device: устройство для модели
        model_file: путь к файлу модели
        model_type: тип архитектуры
        use_optimization: использовать ли оптимизации для инференса
    """
    key = (str(device), model_file, model_type, use_optimization)
    
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    
    if use_optimization:
        model = create_optimized_model(device, model_type)
    else:
        model = create_model(device, model_type)
    
    # Загружаем веса
    import torch
    model.load_state_dict(torch.load(model_file, map_location=device))
    model.to(device)
    model.eval()
    
    _MODEL_CACHE[key] = model
    return model

def get_model_info(model_type='mobilenet_v3_small'):
    """
    Возвращает информацию о модели для выбора оптимальной архитектуры.
    """
    model_info = {
        'mobilenet_v3_small': {
            'speed': 'fastest',
            'accuracy': 'good',
            'params': '~2.5M',
            'description': 'Оптимален для скорости инференса'
        },
        'efficientnet_b0': {
            'speed': 'fast',
            'accuracy': 'better',
            'params': '~5.3M',
            'description': 'Хороший баланс скорости и качества'
        },
        'resnet18': {
            'speed': 'medium',
            'accuracy': 'best',
            'params': '~11.7M',
            'description': 'Лучшее качество, но медленнее'
        },
        'mobilenet_v2': {
            'speed': 'fast',
            'accuracy': 'good',
            'params': '~3.5M',
            'description': 'Альтернатива MobileNetV3'
        }
    }
    
    return model_info.get(model_type, model_info['mobilenet_v3_small']) 
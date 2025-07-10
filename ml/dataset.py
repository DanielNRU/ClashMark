import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import os
from torchvision import transforms
import logging

logger = logging.getLogger(__name__)

class CollisionImageDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform
        
        # Проверяем наличие необходимых колонок
        if 'image_file' not in self.dataframe.columns:
            raise ValueError("Колонка 'image_file' не найдена в DataFrame")
        
        # Фильтруем строки с валидными путями к изображениям
        self.dataframe = self.dataframe[
            self.dataframe['image_file'].notna() & 
            (self.dataframe['image_file'] != '') &
            self.dataframe['image_file'].astype(str).str.strip() != ''
        ].copy()
        
        if len(self.dataframe) == 0:
            raise ValueError("Не найдено валидных изображений для обучения")
        
        logger.info(f"Загружено {len(self.dataframe)} изображений для обучения")
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        image_path = row['image_file']
        # Определяем метку и приводим к float
        if 'status' in row:
            label = row['status']
        elif 'IsResolved' in row:
            label = row['IsResolved']
        else:
            label = -1  # -1 используется для visual/Reviewed
        try:
            label = float(label)
        except Exception:
            label = -1.0
        try:
            # Проверяем существование файла
            if not os.path.exists(image_path):
                logger.warning(f"Файл не найден: {image_path}")
                image = torch.zeros(3, 224, 224)
            else:
                image = Image.open(image_path).convert('RGB')
                if self.transform:
                    image = self.transform(image)
        except Exception as e:
            logger.warning(f"Ошибка загрузки изображения {image_path}: {e}")
            image = torch.zeros(3, 224, 224)
        return image, torch.tensor(label, dtype=torch.float32)

def create_transforms(is_training=True):
    if is_training:
        return transforms.Compose([
            transforms.Resize((500, 500)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((500, 500)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]) 
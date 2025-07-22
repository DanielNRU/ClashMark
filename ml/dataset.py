import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import os
from torchvision import transforms
import logging
from core.image_utils import find_image_by_name, get_relative_image_path, get_absolute_image_path_optimized

logger = logging.getLogger(__name__)

class CollisionImageDataset(Dataset):
    """
    Класс-обёртка для работы с изображениями коллизий в формате PyTorch Dataset.
    Позволяет удобно загружать изображения и метки для обучения и инференса.
    """
    def __init__(self, dataframe, transform=None, session_dir=None):
        self.dataframe = dataframe.copy()
        self.transform = transform
        self.session_dir = session_dir
        # Проверяем, что есть нужные колонки и сопоставляем пути к изображениям
        if 'image_href' in self.dataframe.columns and session_dir:
            def robust_find(href):
                # Сначала ищем по оптимизированному пути, если не найдено — ищем по имени
                path = get_absolute_image_path_optimized(href, session_dir) if href else None
                if not path:
                    path = find_image_by_name(href, session_dir) if href else None
                    if not path:
                        rel_path = get_relative_image_path(href) if href else ''
                        logging.warning(f"[CollisionImageDataset] Не найдено изображение для href: {href} | session_dir: {session_dir} | rel_path: {rel_path}")
                return path
            self.dataframe['image_file'] = self.dataframe['image_href'].apply(robust_find)
        if 'image_file' not in self.dataframe.columns:
            raise ValueError("Колонка 'image_file' не найдена в DataFrame")
        # Оставляем только строки с валидными путями к изображениям
        self.dataframe = self.dataframe[
            self.dataframe['image_file'].notna() &
            (self.dataframe['image_file'] != '') &
            self.dataframe['image_file'].astype(str).str.strip() != ''
        ].copy()
        # Если после фильтрации не осталось изображений — выбрасываем ошибку
        if len(self.dataframe) == 0:
            raise ValueError(f"Не найдено валидных изображений для обучения!\nВременная папка: {session_dir}")
        logger.info(f"Загружено {len(self.dataframe)} изображений для обучения")

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        image_path = row['image_file']
        # Определяем метку для обучения/инференса (0 — разрешено, 1 — активно)
        if 'IsResolved' in row and not pd.isna(row['IsResolved']):
            label = row['IsResolved']
        elif 'label' in row and not pd.isna(row['label']):
            label = row['label']
        elif 'status' in row and not pd.isna(row['status']):
            label = row['status']
        else:
            raise ValueError(f"В строке нет метки (IsResolved, label, status): {row}")
        try:
            label = float(label)
        except Exception:
            raise ValueError(f"Метка не преобразуется в float: {label} (row: {row})")
        try:
            # Проверяем существование файла изображения
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
    """
    Возвращает пайплайн аугментаций для изображений.
    Для обучения — с аугментациями, для инференса — только resize и нормализация.
    """
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
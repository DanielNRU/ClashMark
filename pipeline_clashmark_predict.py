#!/usr/bin/env python3
"""
Инференс-пайплайн для получения предсказаний по коллизиям BIM-моделей
Использует обученную модель и сохраняет результат в XML
"""

import logging
import torch
import sys
import yaml
import lxml.etree as ET
import os
import pandas as pd
from torchvision import models, transforms
import numpy as np
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_all_category_pairs(yaml_path):
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    def to_pair_set(section):
        pairs = set()
        for pair in data.get(section, []):
            if isinstance(pair, list) and len(pair) == 2:
                a, b = pair
                if a <= b:
                    pairs.add((a, b))
                else:
                    pairs.add((b, a))
        return pairs
    return {
        'can': to_pair_set('can'),
        'cannot': to_pair_set('cannot'),
        'visual': to_pair_set('visual'),
    }

def get_pair(row):
    a = row.get('element1_category', '')
    b = row.get('element2_category', '')
    return (a, b) if a <= b else (b, a)

def parse_xml_data(xml_path):
    logger.info(f"Парсинг XML файла: {xml_path}")
    tree = ET.parse(xml_path)
    root = tree.getroot()
    clashes = []
    for clash_result in root.findall('.//clashresult'):
        clash = {}
        clash['clash_id'] = clash_result.get('guid', '')
        clash['clash_name'] = clash_result.get('name', '')
        clash['status'] = clash_result.get('status', '')
        clash['distance'] = float(clash_result.get('distance', 0))
        image_href = clash_result.get('href', '')
        clash['image_href'] = image_href.replace('\\', '/').strip() if image_href else ''
        cp = clash_result.find('.//clashpoint/pos3f')
        clash['clash_x'] = float(cp.get('x', 0)) if cp is not None else 0
        clash['clash_y'] = float(cp.get('y', 0)) if cp is not None else 0
        clash['clash_z'] = float(cp.get('z', 0)) if cp is not None else 0
        grid = clash_result.find('.//gridlocation')
        clash['grid_location'] = grid.text if grid is not None else ''
        rs = clash_result.find('.//resultstatus')
        clash['resultstatus'] = rs.text.strip() if rs is not None and rs.text else ''
        if clash['resultstatus'] == 'Активн.':
            clash['IsResolved'] = 1
        elif clash['resultstatus'] == 'Подтверждено':
            clash['IsResolved'] = 0
        else:
            clash['IsResolved'] = -1
        objs = clash_result.findall('.//clashobject')
        for i, prefix in zip([0, 1], ['element1', 'element2']):
            if len(objs) > i:
                obj = objs[i]
                path_nodes = obj.findall('.//pathlink/node')
                path_parts = [n.text for n in path_nodes if n.text]
                clash[f'{prefix}_category'] = path_parts[3] if len(path_parts) > 3 else ''
                clash[f'{prefix}_family'] = path_parts[4] if len(path_parts) > 4 else ''
                clash[f'{prefix}_type'] = path_parts[5] if len(path_parts) > 5 else ''
        clashes.append(clash)
    df = pd.DataFrame(clashes)
    logger.info(f"Извлечено {len(df)} коллизий из XML")
    return df

def extract_images_dir_from_href(image_href):
    if not image_href:
        return ''
    clean_path = image_href.replace('\\', '/').strip()
    dir_path = os.path.dirname(clean_path)
    return "uploads/" + dir_path

def find_image_by_href(image_href, images_dir):
    if not image_href:
        return None
    clean_path = image_href.replace('\\', '/').strip()
    filename = os.path.basename(clean_path)
    full_path = os.path.join(images_dir, filename)
    if os.path.exists(full_path):
        return full_path
    logger.debug(f"Файл не найден: {full_path}")
    return None

def collect_dataset_from_multiple_files(xml_paths):
    logger.info(f"Сбор датасета из {len(xml_paths)} XML файлов...")
    all_dataframes = []
    for xml_path in xml_paths:
        try:
            df = parse_xml_data(xml_path)
            images_dir = ''
            if len(df) > 0 and 'image_href' in df.columns:
                first_href = df['image_href'].iloc[0]
                if first_href:
                    images_dir = extract_images_dir_from_href(first_href)
                    logger.info(f"Определена папка с изображениями: {images_dir}")
            if not images_dir:
                logger.warning(f"Не удалось определить папку с изображениями для {xml_path}")
                continue
            df['image_file'] = df['image_href'].apply(lambda href: find_image_by_href(href, images_dir))
            df['source_file'] = os.path.basename(xml_path)
            all_dataframes.append(df)
        except Exception as e:
            logger.error(f"Ошибка при обработке файла {xml_path}: {e}")
            continue
    if not all_dataframes:
        logger.error("Не удалось обработать ни одного XML файла")
        return pd.DataFrame()
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    df_with_images = combined_df[combined_df['image_file'].notna() & combined_df['image_file'].apply(lambda x: x is not None)]
    logger.info(f"Найдено {len(df_with_images)} коллизий с изображениями из {len(combined_df)} общих")
    return df_with_images

def main():
    # Пути к файлам
    xml_paths = ["uploads/ОВ-Все.xml"]  # Замените на нужные XML-файлы
    model_path = "model/model_clashmark.pt"
    category_pairs_path = "category_pairs.yaml"

    for xml_path in xml_paths:
        # Формируем имя выходного файла на основе исходного
        base_name = os.path.splitext(os.path.basename(xml_path))[0]
        output_xml = f"cv_results_{base_name}.xml"

        # Загружаем все пары
        pairs = load_all_category_pairs(category_pairs_path)
        can_pairs = pairs['can']
        cannot_pairs = pairs['cannot']
        visual_pairs = pairs['visual']

        # Определяем устройство
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Используется устройство: {device}")

        try:
            # Собираем датасет
            df = collect_dataset_from_multiple_files([xml_path])
            if len(df) == 0:
                logger.error("Не найдено изображений для анализа")
                continue

            # Загружаем модель
            model = create_model(device)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            model.eval()
            logger.info(f"Модель загружена из {model_path}")

            # Для каждой строки определяем тип пары и выставляем предсказание
            predictions = []
            confidences = []
            visual_rows = []
            visual_indices = []
            count_can = 0
            count_cannot = 0
            count_visual = 0
            for idx, row in df.iterrows():
                pair = get_pair(row)
                if pair in can_pairs:
                    predictions.append(0)
                    confidences.append(1.0)
                    count_can += 1
                elif pair in cannot_pairs:
                    predictions.append(1)
                    confidences.append(1.0)
                    count_cannot += 1
                elif pair in visual_pairs:
                    visual_rows.append(row)
                    visual_indices.append(idx)
                    predictions.append(None)
                    confidences.append(None)
                    count_visual += 1
                else:
                    visual_rows.append(row)
                    visual_indices.append(idx)
                    predictions.append(None)
                    confidences.append(None)
                    count_visual += 1

            logger.info(f"Коллизий размечено по category_pairs.yaml: {count_can + count_cannot}")
            logger.info(f"Коллизий отправлено на разметку моделью: {count_visual}")

            # Инференс только для visual
            if visual_rows:
                visual_df = pd.DataFrame(visual_rows)
                transform = create_transforms(is_training=False)
                visual_pred_df = predict(model, device, visual_df, transform)
                # Статистика по классам
                stat = visual_pred_df['cv_prediction'].value_counts().to_dict()
                logger.info("Статистика по разметке моделью:")
                for k in sorted(stat):
                    name = {1: 'Критичные', 0: 'Некритичные', -1: 'Требуют анализа'}.get(k, str(k))
                    logger.info(f"  {name} ({k}): {stat[k]}")
                for i, idx in enumerate(visual_indices):
                    predictions[idx] = int(visual_pred_df.iloc[i]['cv_prediction'])
                    confidences[idx] = float(visual_pred_df.iloc[i]['cv_confidence'])

            # Записываем предсказания в DataFrame
            df['cv_prediction'] = predictions
            df['cv_confidence'] = confidences

            # Сохраняем результат в XML
            export_to_xml(df, output_xml, original_xml_path=xml_path)
            logger.info(f"Результаты инференса сохранены в {output_xml}")

        except Exception as e:
            logger.error(f"Ошибка в инференс-пайплайне: {e}")
            continue

def export_to_xml(df: pd.DataFrame, output_xml_path: str, original_xml_path: str = None):
    """Экспортирует результаты в XML формат с обновленными полями (использует lxml)"""
    logger.info(f"Экспорт результатов в XML: {output_xml_path}")
    try:
        # Если указан оригинальный XML, используем его как шаблон
        if original_xml_path and os.path.exists(original_xml_path):
            tree = ET.parse(original_xml_path)
            root = tree.getroot()
        else:
            # Создаем новый XML
            root = ET.Element("clashresults")
        # Создаем словарь для быстрого поиска по clash_id
        df_dict = df.set_index('clash_id').to_dict('index')
        # Обновляем существующие записи или создаем новые
        for clash_id, row_data in df_dict.items():
            # Ищем существующий clashresult
            clash_result = root.find(f".//clashresult[@guid='{clash_id}']")
            if clash_result is None:
                # Создаем новый clashresult
                clash_result = ET.SubElement(root, "clashresult")
                clash_result.set('guid', clash_id)
                clash_result.set('name', row_data.get('clash_name', ''))
            # Обновляем атрибуты
            clash_result.set('status', row_data.get('status', ''))
            clash_result.set('distance', str(row_data.get('distance', 0)))
            # Обновляем resultstatus
            resultstatus_elem = clash_result.find('.//resultstatus')
            if resultstatus_elem is None:
                resultstatus_elem = ET.SubElement(clash_result, "resultstatus")
            resultstatus_elem.text = row_data.get('resultstatus', '')
            # Добавляем информацию о предсказании
            prediction_elem = clash_result.find('.//cv_prediction')
            if prediction_elem is None:
                prediction_elem = ET.SubElement(clash_result, "cv_prediction")
            prediction_elem.text = str(row_data.get('cv_prediction', ''))
            confidence_elem = clash_result.find('.//cv_confidence')
            if confidence_elem is None:
                confidence_elem = ET.SubElement(clash_result, "cv_confidence")
            confidence_elem.text = str(row_data.get('cv_confidence', ''))
        # Сохраняем XML с pretty_print
        with open(output_xml_path, 'wb') as f:
            f.write(ET.tostring(root, encoding='utf-8', xml_declaration=True, pretty_print=True))
        logger.info(f"XML экспортирован успешно: {output_xml_path}")
        logger.info(f"Обработано {len(df_dict)} записей")
    except Exception as e:
        logger.error(f"Ошибка при экспорте в XML: {e}")
        raise

# --- DATASET ---
class CollisionImageDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform
    def __len__(self):
        return len(self.dataframe)
    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        image_path = row['image_file']
        label = row.get('IsResolved', -1)
        try:
            image = Image.open(image_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            logger.error(f"Ошибка загрузки изображения {image_path}: {e}")
            image = torch.zeros(3, 224, 224)
        return image, torch.tensor(label, dtype=torch.float32)

# --- MODEL ---
def create_model(device):
    logger.info("Создание модели...")
    model = models.mobilenet_v3_small(weights='IMAGENET1K_V1')
    model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, 1)
    model = model.to(device)
    logger.info("Модель создана успешно")
    return model

def create_transforms(is_training=True):
    if is_training:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

# --- INFERENCE ---
def predict(model, device, df, transform, batch_size=16, confidence_threshold=0.5, low_confidence_threshold=0.3, high_confidence_threshold=0.7):
    logger.info(f"Предсказания для {len(df)} изображений")
    dataset = CollisionImageDataset(df, transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0 if device.type == 'cuda' else 2)
    predictions = []
    confidences = []
    model.eval()
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            outputs = model(images).squeeze()
            probs = torch.sigmoid(outputs)
            batch_confidences = probs.cpu().numpy()
            batch_confidences = np.atleast_1d(batch_confidences)
            batch_predictions = []
            for prob in batch_confidences:
                if prob >= high_confidence_threshold:
                    batch_predictions.append(1)
                elif prob <= low_confidence_threshold:
                    batch_predictions.append(0)
                else:
                    batch_predictions.append(-1)
            predictions.extend(batch_predictions)
            confidences.extend(batch_confidences)
    df_result = df.copy()
    df_result['cv_prediction'] = predictions
    df_result['cv_confidence'] = confidences
    df_result = fill_xml_fields(df_result)
    return df_result

def fill_xml_fields(df):
    df_result = df.copy()
    for idx, row in df_result.iterrows():
        prediction = row['cv_prediction']
        if prediction == 0:
            df_result.at[idx, 'IsResolved'] = 0
            df_result.at[idx, 'status'] = 'approved'
            df_result.at[idx, 'resultstatus'] = 'Подтверждено'
        elif prediction == 1:
            df_result.at[idx, 'IsResolved'] = 1
            df_result.at[idx, 'status'] = 'active'
            df_result.at[idx, 'resultstatus'] = 'Активн.'
        elif prediction == -1:
            df_result.at[idx, 'IsResolved'] = -1
            df_result.at[idx, 'status'] = 'reviewed'
            df_result.at[idx, 'resultstatus'] = 'Проанализировано'
    return df_result

if __name__ == "__main__":
    main() 
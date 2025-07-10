import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from PIL import Image
import os
from ml.model import create_model
from ml.dataset import CollisionImageDataset, create_transforms
from core.xml_utils import parse_xml_data, load_all_category_pairs, get_pair, export_to_xml, export_to_bimstep_xml
from core.image_utils import find_image_by_name

def fill_xml_fields(df):
    df_result = df.copy()
    for idx, row in df_result.iterrows():
        prediction = row['cv_prediction']
        if prediction == 0:
            # Approved (can)
            df_result.at[idx, 'IsResolved'] = 0
            df_result.at[idx, 'status'] = 'approved'
            df_result.at[idx, 'resultstatus'] = 'Подтверждено'
        elif prediction == 1:
            # Active (cannot)
            df_result.at[idx, 'IsResolved'] = 1
            df_result.at[idx, 'status'] = 'active'
            df_result.at[idx, 'resultstatus'] = 'Активн.'
        elif prediction == -1:
            # Reviewed (visual)
            df_result.at[idx, 'IsResolved'] = -1
            df_result.at[idx, 'status'] = 'reviewed'
            df_result.at[idx, 'resultstatus'] = 'Проанализировано'
    return df_result

def predict(model, device, df, transform, batch_size=16, confidence_threshold=0.5, low_confidence_threshold=0.3, high_confidence_threshold=0.7):
    import torch
    if isinstance(device, str):
        device = torch.device(device)
    dataset = CollisionImageDataset(df, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
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
                    # Высокая уверенность -> Active (cannot)
                    batch_predictions.append(1)
                elif prob <= low_confidence_threshold:
                    # Низкая уверенность -> Approved (can)
                    batch_predictions.append(0)
                else:
                    # Средняя уверенность -> Reviewed (visual)
                    batch_predictions.append(-1)
            predictions.extend(batch_predictions)
            confidences.extend(batch_confidences)
    df_result = df.copy()
    df_result['cv_prediction'] = predictions
    df_result['cv_confidence'] = confidences
    df_result = fill_xml_fields(df_result)
    return df_result

# --- Сбор датасета из XML и изображений ---
def collect_dataset_from_multiple_files(xml_paths, images_dir=None, export_format='standard'):
    all_dataframes = []
    for xml_path in xml_paths:
        df = parse_xml_data(xml_path, export_format=export_format)
        if len(df) == 0:
            continue
        if not images_dir:
            continue
        df['image_file'] = df['image_href'].apply(lambda href: find_image_by_name(href, images_dir) if href else None)
        df['source_file'] = os.path.basename(xml_path)
        all_dataframes.append(df)
    if not all_dataframes:
        return pd.DataFrame()
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    df_with_images = combined_df[combined_df['image_file'].notna() & combined_df['image_file'].apply(lambda x: x is not None)]
    return df_with_images 
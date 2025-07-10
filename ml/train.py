import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from ml.model import create_model
from ml.dataset import CollisionImageDataset, create_transforms
from core.xml_utils import parse_xml_data, load_all_category_pairs, get_pair
from core.image_utils import find_image_by_name
import os
import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split

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
    if not isinstance(df_with_images, pd.DataFrame):
        df_with_images = pd.DataFrame(df_with_images)
    # Оставляем только классы 0 и 1
    if 'IsResolved' in df_with_images.columns:
        df_with_images = df_with_images[df_with_images['IsResolved'].isin([0, 1])].copy()
    return df_with_images

def train_model(df, epochs=10, batch_size=16, learning_rate=1e-4, device=None, progress_callback=None):
    # --- Фильтрация только по двум классам ---
    df = df[df['IsResolved'].isin([0, 1])].copy()
    
    # Проверяем, что у нас есть оба класса
    if len(df['IsResolved'].unique()) < 2:
        return None
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Разделяем на train и validation
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['IsResolved'])
    
    model = create_model(device)
    transform = create_transforms(is_training=True)
    
    train_dataset = CollisionImageDataset(train_df, transform)
    val_dataset = CollisionImageDataset(val_df, transform)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Инициализируем массивы для метрик
    train_losses = []
    val_losses = []
    val_accuracies = []
    val_f1s = []
    val_recalls = []
    val_precisions = []
    
    for epoch in range(epochs):
        # Обучение
        model.train()
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_dataloader):
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            if progress_callback:
                progress_callback(epoch, i, len(train_dataloader), running_loss/(i+1), None, None, None, None)
        
        train_loss = running_loss / len(train_dataloader)
        train_losses.append(train_loss)
        
        # Валидация
        model.eval()
        val_loss = 0.0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in val_dataloader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels.unsqueeze(1))
                val_loss += loss.item()
                
                predictions = (torch.sigmoid(outputs) > 0.5).float().squeeze()
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        val_loss = val_loss / len(val_dataloader)
        val_losses.append(val_loss)
        
        # Вычисляем метрики
        val_accuracy = accuracy_score(all_labels, all_predictions)
        val_f1 = f1_score(all_labels, all_predictions, zero_division='warn')
        val_recall = recall_score(all_labels, all_predictions, zero_division='warn')
        val_precision = precision_score(all_labels, all_predictions, zero_division='warn')
        
        val_accuracies.append(val_accuracy)
        val_f1s.append(val_f1)
        val_recalls.append(val_recall)
        val_precisions.append(val_precision)
        
        if progress_callback:
            progress_callback(epoch, len(train_dataloader)-1, len(train_dataloader), train_loss, val_loss, val_accuracy, val_f1, val_recall, val_precision)
    
    # Сохраняем модель
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"model_{timestamp}.pt"
    model_path = os.path.join('model', model_filename)
    os.makedirs('model', exist_ok=True)
    torch.save(model.state_dict(), model_path)
    
    # Вычисляем финальные метрики
    final_accuracy = val_accuracies[-1] if val_accuracies else 0
    final_f1 = val_f1s[-1] if val_f1s else 0
    final_recall = val_recalls[-1] if val_recalls else 0
    final_precision = val_precisions[-1] if val_precisions else 0
    
    # Создаём confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    
    metrics = {
        'final_accuracy': final_accuracy,
        'final_f1': final_f1,
        'final_recall': final_recall,
        'final_precision': final_precision,
        'confusion_matrix': cm.tolist(),
        'val_precisions': val_precisions,
        'val_f1s': val_f1s,
        'val_recalls': val_recalls,
        'val_accuracies': val_accuracies,
        'train_losses': train_losses,
        'val_losses': val_losses
    }
    
    return metrics 
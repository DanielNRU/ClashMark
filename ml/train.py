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
from torchvision import transforms

# --- Сбор датасета из XML и изображений ---
def collect_dataset_from_multiple_files(xml_paths, images_dir=None, export_format='standard'):
    all_dataframes = []
    for xml_path in xml_paths:
        df = parse_xml_data(xml_path, export_format=export_format)
        if len(df) == 0:
            continue
        if not images_dir:
            continue
        # Гарантируем, что df — DataFrame
        if not isinstance(df, pd.DataFrame):
            print(f"[DEBUG] Преобразую df в DataFrame после parse_xml_data, type(df)={type(df)}")
            df = pd.DataFrame(df)
        df['image_file'] = df['image_href'].apply(lambda href: find_image_by_name(href, images_dir) if href else None)
        df['source_file'] = os.path.basename(xml_path)
        all_dataframes.append(df)
    if not all_dataframes:
        return pd.DataFrame()
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    # Гарантируем, что combined_df — DataFrame
    if not isinstance(combined_df, pd.DataFrame):
        print(f"[DEBUG] Преобразую combined_df в DataFrame после concat, type(combined_df)={type(combined_df)}")
        combined_df = pd.DataFrame(combined_df)
    df_with_images = combined_df[combined_df['image_file'].notna() & combined_df['image_file'].apply(lambda x: x is not None)]
    if not isinstance(df_with_images, pd.DataFrame):
        print(f"[DEBUG] Преобразую df_with_images в DataFrame после фильтрации, type(df_with_images)={type(df_with_images)}")
        df_with_images = pd.DataFrame(df_with_images)
    # Оставляем только классы 0 и 1
    if 'IsResolved' in df_with_images.columns:
        df_with_images = df_with_images[df_with_images['IsResolved'].isin([0, 1])].copy()
        if not isinstance(df_with_images, pd.DataFrame):
            print(f"[DEBUG] Преобразую df_with_images в DataFrame после фильтрации IsResolved, type(df_with_images)={type(df_with_images)}")
            df_with_images = pd.DataFrame(df_with_images)
    # --- Логирование распределения классов ---
    import logging
    logger = logging.getLogger(__name__)
    if isinstance(df_with_images, pd.DataFrame) and 'IsResolved' in df_with_images.columns:
        class_counts = df_with_images['IsResolved'].value_counts().to_dict()
    else:
        print(f"[DEBUG] df_with_images не DataFrame или нет колонки IsResolved, type(df_with_images)={type(df_with_images)}")
        class_counts = {}
    logger.info(f"Распределение классов после фильтрации: {class_counts}")
    print(f"[DEBUG] Распределение классов после фильтрации: {class_counts}")
    return df_with_images

def create_transforms(is_training=True):
    if is_training:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            transforms.ToTensor(),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

def train_model(df, epochs=10, batch_size=16, learning_rate=1e-4, device=None, progress_callback=None, model_filename=None):
    # --- Фильтрация только по двум классам ---
    df = df[df['IsResolved'].isin([0, 1])].copy()
    
    # Проверяем, что у нас есть оба класса
    if len(df['IsResolved'].unique()) < 2:
        return None
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Разделяем на train и validation
    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df['IsResolved']
    )
    # --- Вывод распределения классов ---
    if isinstance(train_df, pd.DataFrame) and 'IsResolved' in train_df.columns:
        print(f"[DEBUG] Train class distribution: {train_df['IsResolved'].value_counts().to_dict()}")
    else:
        print(f"[DEBUG] train_df не DataFrame или нет колонки IsResolved, type(train_df)={type(train_df)}")
    if isinstance(val_df, pd.DataFrame) and 'IsResolved' in val_df.columns:
        print(f"[DEBUG] Val class distribution: {val_df['IsResolved'].value_counts().to_dict()}")
    else:
        print(f"[DEBUG] val_df не DataFrame или нет колонки IsResolved, type(val_df)={type(val_df)}")
    
    model = create_model(device)
    transform = create_transforms(is_training=True)
    
    train_dataset = CollisionImageDataset(train_df, transform)
    val_dataset = CollisionImageDataset(val_df, transform)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # --- CLASS WEIGHTS ---
    labels = train_df['IsResolved'].values
    n_pos = np.sum(labels == 1)
    n_neg = np.sum(labels == 0)
    if n_pos > 0:
        pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32, device=device)
    else:
        pos_weight = torch.tensor([1.0], dtype=torch.float32, device=device)
    print(f"[DEBUG] pos_weight для BCEWithLogitsLoss: {pos_weight.item():.3f}")
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Инициализируем массивы для метрик
    train_losses = []
    val_losses = []
    val_accuracies = []
    val_f1s = []
    val_recalls = []
    val_precisions = []
    confusion_matrices = []
    
    for epoch in range(epochs):
        # Обучение
        model.train()
        running_loss = 0.0
        all_train_predictions = []
        all_train_labels = []
        for i, (images, labels) in enumerate(train_dataloader):
            print(f"[DEBUG][EPOCH {epoch+1}][BATCH {i+1}] images.shape: {images.shape}, labels: {labels[:5]}")
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            predictions = (torch.sigmoid(outputs) > 0.5).float().flatten()
            all_train_predictions.extend(predictions.detach().cpu().numpy().tolist())
            all_train_labels.extend(labels.detach().cpu().numpy().flatten().tolist())
            if progress_callback:
                print(f"[CALL][progress_callback] epoch={epoch}, batch={i}, update_metrics=False")
                try:
                    progress_callback(
                        epoch, i, len(train_dataloader), running_loss / (i + 1), None, None, None, None, None,
                        update_metrics=False
                    )
                except TypeError:
                    progress_callback(
                        epoch, i, len(train_dataloader), running_loss / (i + 1), None, None, None, None, None
                    )
        # === Метрики по train ===
        if all_train_labels and all_train_predictions and len(set(all_train_labels)) > 1:
            train_accuracy = accuracy_score(all_train_labels, all_train_predictions)
            train_f1 = f1_score(all_train_labels, all_train_predictions, zero_division=0)
            train_recall = recall_score(all_train_labels, all_train_predictions, zero_division=0)
            train_precision = precision_score(all_train_labels, all_train_predictions, zero_division=0)
            try:
                train_conf_matrix = confusion_matrix(all_train_labels, all_train_predictions)
                print(f"[EPOCH {epoch+1}] confusion_matrix={train_conf_matrix.tolist()}")
            except Exception as e:
                train_conf_matrix = None
                print(f"[EPOCH {epoch+1}] Ошибка при вычислении confusion_matrix по train: {e}")
        else:
            print(f"[EPOCH {epoch+1}] Нет данных или только один класс для подсчёта метрик по train")
            train_accuracy = None
            train_f1 = None
            train_recall = None
            train_precision = None
            train_conf_matrix = None
        # === Валидация ===
        model.eval()
        val_loss = 0.0
        val_targets = []
        val_preds = []
        with torch.no_grad():
            for images, labels in val_dataloader:
                images = images.to(device)
                labels = labels.to(device).unsqueeze(1)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                preds = (torch.sigmoid(outputs) > 0.5).float().flatten()
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(labels.cpu().numpy().flatten())
        val_loss /= len(val_dataloader)
        if val_targets and val_preds and len(set(val_targets)) > 1:
            val_accuracy = accuracy_score(val_targets, val_preds)
            val_f1 = f1_score(val_targets, val_preds, zero_division=0)
            val_recall = recall_score(val_targets, val_preds, zero_division=0)
            val_precision = precision_score(val_targets, val_preds, zero_division=0)
            try:
                val_conf_matrix = confusion_matrix(val_targets, val_preds)
                print(f"[EPOCH {epoch+1}] val_confusion_matrix={val_conf_matrix.tolist()}")
            except Exception as e:
                val_conf_matrix = None
                print(f"[EPOCH {epoch+1}] Ошибка при вычислении confusion_matrix по val: {e}")
        else:
            print(f"[EPOCH {epoch+1}] Нет данных или только один класс для подсчёта метрик по val")
            val_accuracy = None
            val_f1 = None
            val_recall = None
            val_precision = None
            val_conf_matrix = None
        # Вызов progress_callback только после валидации (раз в эпоху)
        if progress_callback:
            print(f"[CALL][progress_callback] epoch={epoch}, batch={len(train_dataloader)-1}, update_metrics=True")
            try:
                progress_callback(
                    epoch, len(train_dataloader)-1, len(train_dataloader), running_loss / len(train_dataloader),
                    val_loss, val_accuracy, val_f1, val_recall, val_precision, update_metrics=True
                )
            except TypeError:
                progress_callback(
                    epoch, len(train_dataloader)-1, len(train_dataloader), running_loss / len(train_dataloader),
                    val_loss, val_accuracy, val_f1, val_recall, val_precision
                )
        model.train()
        
        val_losses.append(val_loss)
        
        # Добавляю заполнение всех массивов метрик
        train_losses.append(running_loss / len(train_dataloader))
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        val_f1s.append(val_f1)
        val_recalls.append(val_recall)
        val_precisions.append(val_precision)
        
        # Удалён устаревший блок с all_labels и all_predictions
        # (все метрики уже вычисляются выше на основе val_targets и val_preds)
        #
        # Пример удалённого кода:
        # all_labels = np.array(all_labels).astype(int)
        # ...
        # val_accuracy = accuracy_score(all_labels, all_predictions)
        # ...
        # confusion_matrices.append(val_conf_matrix.tolist())
        # ...
        # После валидации — вызываем progress_callback с метриками
        # (удалён дублирующий вызов ниже)
    
    # Сохраняем модель
    if model_filename is None:
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
    
    # Создаём confusion matrix для последней эпохи
    # Используем последние val_targets и val_preds
    if val_targets and val_preds and len(set(val_targets)) > 1:
        cm = confusion_matrix(val_targets, val_preds)
        cm_list = cm.tolist()
    else:
        cm_list = None
    metrics = {
        'final_accuracy': final_accuracy,
        'final_f1': final_f1,
        'final_recall': final_recall,
        'final_precision': final_precision,
        'confusion_matrix': cm_list,
        'confusion_matrices': confusion_matrices,
        'val_precisions': val_precisions,
        'val_f1s': val_f1s,
        'val_recalls': val_recalls,
        'val_accuracies': val_accuracies,
        'train_losses': train_losses,
        'val_losses': val_losses
    }
    
    return metrics 
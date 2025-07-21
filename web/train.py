from flask import request, render_template, jsonify
import os
import tempfile
import traceback
from web.utils import save_uploaded_files, cleanup_temp_dir, unzip_archives, safe_filename_with_cyrillic, load_settings
from ml.train import collect_dataset_from_multiple_files, train_model
from collections import Counter
from core.image_utils import find_image_by_name
from core.xml_utils import parse_xml_data, get_pair
import pandas as pd
import zipfile
import yaml
from web.progress import update_train_progress, load_train_progress
import datetime

# Глобальная переменная для хранения последних метрик обучения (для live-обновления)
last_train_metrics = {}

def handle_train_request():
    if request.method == 'POST':
        try:
            xml_files = request.files.getlist('xml_file')
            zip_files = request.files.getlist('zip_file')
            batch_size = int(request.form.get('batch_size', 16))
            epochs = int(request.form.get('epochs', 5))
            
            with tempfile.TemporaryDirectory() as temp_dir:
                global last_train_temp_dir
                last_train_temp_dir = temp_dir
                settings = load_settings()
                export_format = settings.get('export_format', 'standard')
                
                # Сохраняем XML файлы
                xml_paths = []
                for xml_file in xml_files:
                    if not xml_file.filename:
                        continue
                    safe_name = safe_filename_with_cyrillic(xml_file.filename)
                    xml_path = os.path.join(temp_dir, safe_name)
                    xml_file.save(xml_path)
                    xml_paths.append(xml_path)
                
                # Сохраняем и распаковываем ZIP файлы
                zip_paths = []
                for zip_file in zip_files:
                    if not zip_file.filename:
                        continue
                    safe_name = safe_filename_with_cyrillic(zip_file.filename)
                    zip_path = os.path.join(temp_dir, safe_name)
                    zip_file.save(zip_path)
                    zip_paths.append(zip_path)
                
                # Распаковываем архивы
                for zip_path in zip_paths:
                    if zipfile.is_zipfile(zip_path):
                        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                            for member in zip_ref.namelist():
                                if member.endswith('/') or member.endswith('\\') or member == '' or member[-1] in ['/', '\\']:
                                    continue
                                safe_member = safe_filename_with_cyrillic(member)
                                target_path = os.path.join(temp_dir, safe_member)
                                # Создаём директорию, если нужно
                                os.makedirs(os.path.dirname(target_path), exist_ok=True)
                                with open(target_path, 'wb') as f:
                                    f.write(zip_ref.read(member))
                
                # Анализируем только XML файлы для статистики
                all_dfs = []
                for xml_path in xml_paths:
                    try:
                        df = parse_xml_data(xml_path, export_format=export_format)
                        if len(df) > 0:
                            all_dfs.append(df)
                    except Exception as e:
                        continue
                
                if not all_dfs:
                    return render_template('train.html', error='Не удалось обработать XML-файлы', stats=None)
                
                df = pd.concat(all_dfs, ignore_index=True)
                df_train = df[df['IsResolved'] != -1]
                # Гарантируем, что df_train — DataFrame
                import numpy as np
                if not isinstance(df_train, pd.DataFrame):
                    df_train = pd.DataFrame(df_train)
                # Фильтруем только visual-пары
                with open('category_pairs.yaml', 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
                visual_pairs = set()
                if 'visual' in data and isinstance(data['visual'], list):
                    for pair in data['visual']:
                        if isinstance(pair, list) and len(pair) == 2:
                            a, b = pair
                            visual_pairs.add((a, b) if a <= b else (b, a))
                
                # --- Новый блок: фильтрация только visual Approved/Active и аугментация по наименьшему классу ---
                from PIL import Image
                from PIL.Image import Transpose
                # Определяем тип пары
                def get_pair(row):
                    return tuple(sorted([str(row['element1_category']), str(row['element2_category'])]))
                df_train['pair_type'] = df_train.apply(
                    lambda row: 'visual' if get_pair(row) in visual_pairs else 'other', axis=1
                )
                if not isinstance(df_train, pd.DataFrame):
                    df_train = pd.DataFrame(df_train)
                df_visual = df_train[(df_train['pair_type'] == 'visual') & (df_train['status'].isin(['Approved', 'Active']))].copy()
                if not isinstance(df_visual, pd.DataFrame):
                    df_visual = pd.DataFrame(df_visual)
                df_visual['label'] = df_visual['status'].apply(lambda x: 0 if x == 'Approved' else (1 if x == 'Active' else None))
                df_visual['image_file'] = df_visual['image_href'].apply(lambda href: find_image_by_name(href, temp_dir) if href else None)
                df_visual = df_visual[df_visual['image_file'].notna() & df_visual['image_file'].apply(lambda x: x is not None)]
                if not isinstance(df_visual, pd.DataFrame):
                    df_visual = pd.DataFrame(df_visual)
                class_counts = df_visual['label'].value_counts().to_dict() if 'label' in df_visual.columns else {}
                if class_counts:
                    min_class = min(class_counts, key=lambda k: class_counts[k])
                    max_class = max(class_counts, key=lambda k: class_counts[k])
                    min_count = class_counts.get(min_class, 0)
                    max_count = class_counts.get(max_class, 0)
                else:
                    min_class = 0
                    max_class = 1
                    min_count = 0
                    max_count = 0
                threshold = 100
                def augment_and_save(row, aug_dir, n_aug=6):
                    orig_path = row['image_file']
                    base_name = os.path.splitext(os.path.basename(orig_path))[0]
                    ext = os.path.splitext(orig_path)[1]
                    img = Image.open(orig_path).convert('RGB')
                    aug_rows = []
                    for i in range(n_aug):
                        aug_img = img.copy()
                        if np.random.rand() > 0.5:
                            aug_img = aug_img.transpose(Transpose.FLIP_LEFT_RIGHT)
                        if np.random.rand() > 0.5:
                            aug_img = aug_img.transpose(Transpose.FLIP_TOP_BOTTOM)
                        angle = np.random.choice([0, 90, 180, 270])
                        if angle != 0:
                            aug_img = aug_img.rotate(angle)
                        aug_name = f"{base_name}_aug{i}{ext}"
                        aug_path = os.path.join(aug_dir, aug_name)
                        aug_img.save(aug_path)
                        new_row = row.copy()
                        new_row['image_file'] = aug_path
                        new_row['image_href'] = aug_name
                        aug_rows.append(new_row)
                    return aug_rows
                AUG_DIR = os.path.join(temp_dir, 'BSImages_aug')
                os.makedirs(AUG_DIR, exist_ok=True)
                augmented_rows = []
                if min_count < threshold and max_count < threshold:
                    for label in [0, 1]:
                        for idx, row in df_visual[df_visual['label'] == label].iterrows():
                            augmented_rows.extend(augment_and_save(row, AUG_DIR, n_aug=6))
                elif min_count < threshold:
                    for idx, row in df_visual[df_visual['label'] == min_class].iterrows():
                        augmented_rows.extend(augment_and_save(row, AUG_DIR, n_aug=6))
                df_visual_aug = pd.concat([df_visual, pd.DataFrame(augmented_rows)], ignore_index=True) if augmented_rows else df_visual
                if not isinstance(df_visual_aug, pd.DataFrame):
                    df_visual_aug = pd.DataFrame(df_visual_aug)
                df_final = df_visual_aug.rename(columns={'label': 'IsResolved'})
                # --- Конец нового блока ---

                # stats и прочее считаем по df_final
                class_counts = dict(Counter(df_final['IsResolved'])) if 'IsResolved' in df_final.columns else {}
                stats = {
                    'total_collisions': len(df),
                    'trainable_collisions': len(df_final),
                    'class_counts': class_counts,
                }
                if len(df_final['IsResolved'].unique()) < 2:
                    return render_template('train.html', error='Ошибка: в обучающей выборке только один класс!', stats=stats)
                
                # Инициализируем прогресс обучения
                update_train_progress({
                    'status': 'start',
                    'epoch': 0,
                    'batch': 0,
                    'total_epochs': epochs,
                    'total_batches': 0,
                    'loss': 0,
                    'metrics': {},
                    'log': '',
                    'started': False
                }, temp_dir)
                
                # Запускаем обучение с callback для прогресса
                def progress_callback(epoch, batch, total_batches, train_loss, val_loss, val_acc, f1, recall, precision=None, update_metrics=True):
                    prog = load_train_progress(temp_dir)
                    # Всегда обновляем текущий батч и эпоху
                    prog['started'] = True
                    prog['epoch'] = epoch + 1
                    prog['batch'] = batch + 1
                    prog['total_batches'] = total_batches
                    prog['loss'] = round(float(train_loss), 3)
                    # Метрики и лог — только если update_metrics=True (раз в эпоху)
                    if update_metrics:
                        metrics = prog.setdefault('metrics', {})
                        for key, val in [
                            ('train_losses', train_loss),
                            ('val_losses', val_loss),
                            ('val_accuracies', val_acc),
                            ('val_f1s', f1),
                            ('val_recalls', recall),
                            ('val_precisions', precision)
                        ]:
                            arr = metrics.setdefault(key, [])
                            arr.append(val if val is not None else 0)
                    else:
                        pass # --- НЕ добавляем метрики в массивы: только статус ---
                    update_train_progress(prog, temp_dir)
                model_name = f"model_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
                metrics = train_model(
                    df_final, epochs=epochs, batch_size=batch_size,
                    progress_callback=progress_callback,
                    model_filename=model_name
                )
                
                if metrics is None:
                    return render_template('train.html', error='Ошибка: в обучающей выборке только один класс!', stats=stats)
                
                # Сохраняем метрики в файл
                import json
                
                # Собираем пары категорий
                pair_counts = Counter([get_pair(row) for _, row in df.iterrows()])
                pairs_with_counts = [[p[0], p[1], c] for p, c in pair_counts.items()]
                
                stats_data = {
                    'model_file': model_name,
                    'train_time': datetime.datetime.now().isoformat(),
                    'metrics': metrics,
                    'category_pairs': pairs_with_counts,
                    'logs': load_train_progress(temp_dir).get('log', ''),
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'model_type': request.form.get('model_type', 'mobilenet_v3_small')
                }
                
                # Сохраняем в _stats.json
                stats_path = os.path.join('model', f"{model_name}_stats.json")
                os.makedirs('model', exist_ok=True)
                with open(stats_path, 'w', encoding='utf-8') as f:
                    json.dump(stats_data, f, ensure_ascii=False, indent=2, default=str)
                
                # --- Запись в model_train_log.json ---
                log_path = os.path.join('model', 'model_train_log.json')
                log_entry = {
                    "train_time": stats_data['train_time'],
                    "model_file": stats_data['model_file'],
                    "final_accuracy": stats_data['metrics'].get('final_accuracy'),
                    "final_f1": stats_data['metrics'].get('final_f1'),
                    "final_recall": stats_data['metrics'].get('final_recall'),
                    "final_precision": stats_data['metrics'].get('final_precision'),
                    "confusion_matrix": stats_data['metrics'].get('confusion_matrix'),
                    "epochs": stats_data['epochs'],
                    "batch_size": stats_data['batch_size'],
                    "model_type": stats_data['model_type']
                }
                if os.path.exists(log_path):
                    with open(log_path, 'r', encoding='utf-8') as f:
                        logs = json.load(f)
                    if not isinstance(logs, list):
                        logs = [logs]
                else:
                    logs = []
                logs.append(log_entry)
                with open(log_path, 'w', encoding='utf-8') as f:
                    json.dump(logs, f, ensure_ascii=False, indent=2)
                # --- Конец записи в лог ---
                
                return render_template('train_progress.html')
                
        except Exception as e:
            update_train_progress({
                'status': 'error',
                'log': f'Ошибка обучения: {e}'
            }, temp_dir)
            return render_template('train.html', error=f'Ошибка обучения: {e}\n{traceback.format_exc()}', stats=None)
    
    return render_template('train.html', stats=None)

# Endpoint для live-обновления прогресса обучения
from flask import Blueprint, request
train_bp = Blueprint('train_progress', __name__)

# Глобальная переменная для хранения пути к последнему temp_dir
last_train_temp_dir = None

@train_bp.route('/api/train_progress')
def train_progress():
    global last_train_temp_dir
    if not last_train_temp_dir:
        return jsonify({'status': 'not_started'})
    prog = load_train_progress(last_train_temp_dir)
    return jsonify(prog)

def process_training(xml_files, zip_files, batch_size, epochs):
    # Здесь будет логика запуска обучения модели
    # Возвращать результат, логи, ошибки
    pass

def get_train_logs():
    # Здесь можно реализовать получение логов обучения для live-обновления
    pass 
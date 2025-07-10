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
from web.progress import update_train_progress, train_progress
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
                
                # Фильтруем только visual-пары
                with open('category_pairs.yaml', 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
                visual_pairs = set()
                if 'visual' in data and isinstance(data['visual'], list):
                    for pair in data['visual']:
                        if isinstance(pair, list) and len(pair) == 2:
                            a, b = pair
                            visual_pairs.add((a, b) if a <= b else (b, a))
                
                pair_counts = Counter([get_pair(row) for _, row in df_train.iterrows() if get_pair(row) in visual_pairs])
                pairs_with_counts = sorted([[p[0], p[1], c] for p, c in pair_counts.items()], key=lambda x: (-int(x[2] or 0), str(x[0]), str(x[1])))
                
                trainable_collisions = sum([int(x[2] or 0) for x in pairs_with_counts])
                class_counts = dict(Counter(df_train['IsResolved'])) if 'IsResolved' in df_train.columns else {}
                
                stats = {
                    'total_collisions': len(df),
                    'trainable_collisions': trainable_collisions,
                    'class_counts': class_counts,
                }
                
                if trainable_collisions == 0:
                    return render_template('train.html', error='Не найдено коллизий для обучения. Проверьте, что имена файлов в XML совпадают с именами изображений в архиве, и что архив содержит папку BSImages или изображения в корне.', stats=stats)
                
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
                })
                
                # Запускаем обучение с callback для прогресса
                def progress_callback(epoch, batch, total_batches, train_loss, val_loss, val_acc, f1, recall, precision=None):
                    # Обновляем прогресс только в конце эпохи
                    if batch == total_batches - 1 and val_loss is not None:
                        update_train_progress({
                            'started': True,
                            'epoch': epoch + 1,
                            'batch': batch + 1,
                            'total_batches': total_batches,
                            'loss': round(float(train_loss), 3),
                            'metrics': {
                                'last_train_loss': round(float(train_loss), 3),
                                'last_val_loss': round(float(val_loss), 3) if val_loss is not None else '',
                                'last_val_acc': round(float(val_acc), 3) if val_acc is not None else '',
                                'last_f1': round(float(f1), 3) if f1 is not None else '',
                                'last_recall': round(float(recall), 3) if recall is not None else '',
                                'last_precision': round(float(precision), 3) if precision is not None else ''
                            }
                        })
                        
                        # Добавляем в лог
                        if val_loss is not None and val_acc is not None:
                            f1_str = f"{round(f1,3):.3f}" if f1 is not None else '—'
                            recall_str = f"{round(recall,3):.3f}" if recall is not None else '—'
                            precision_str = f"{round(precision,3):.3f}" if precision is not None else '—'
                            log_entry = f"<div style='margin:8px 0;padding:8px 12px;background:#f8f9fa;border-radius:8px;display:inline-block'>"
                            log_entry += f"<b>Эпоха {epoch+1}:</b> "
                            log_entry += f"Train Loss: <span style='color:#4caf50'>{round(train_loss,3):.3f}</span> | "
                            log_entry += f"Val Loss: <span style='color:#2196f3'>{round(val_loss,3):.3f}</span> | "
                            log_entry += f"Val Accuracy: <span style='color:#ff9800'>{round(val_acc,3):.3f}</span> | "
                            log_entry += f"F1: <span style='color:#9c27b0'>{f1_str}</span> | "
                            log_entry += f"Recall: <span style='color:#607d8b'>{recall_str}</span> | "
                            log_entry += f"Precision: <span style='color:#009688'>{precision_str}</span>"
                            log_entry += "</div><br>"
                            
                            update_train_progress({'log': log_entry})
                
                metrics = train_model(df, epochs=epochs, batch_size=batch_size, progress_callback=progress_callback)
                
                if metrics is None:
                    return render_template('train.html', error='Ошибка: в обучающей выборке только один класс!', stats=stats)
                
                # Сохраняем метрики в файл
                import json
                
                # Собираем пары категорий
                pair_counts = Counter([get_pair(row) for _, row in df.iterrows()])
                pairs_with_counts = [[p[0], p[1], c] for p, c in pair_counts.items()]
                
                stats_data = {
                    'model_file': f"model_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pt",
                    'train_time': datetime.datetime.now().isoformat(),
                    'metrics': metrics,
                    'category_pairs': pairs_with_counts,
                    'logs': train_progress.get('log', ''),
                    'epochs': epochs,
                    'batch_size': batch_size
                }
                
                # Сохраняем в _stats.json
                stats_path = os.path.join('model', f"{stats_data['model_file']}_stats.json")
                os.makedirs('model', exist_ok=True)
                with open(stats_path, 'w', encoding='utf-8') as f:
                    json.dump(stats_data, f, ensure_ascii=False, indent=2, default=str)
                
                # Обновляем финальные метрики
                update_train_progress({
                    'status': 'done',
                    'metrics': {
                        'final_accuracy': metrics.get('final_accuracy', 0),
                        'final_f1': metrics.get('final_f1', 0),
                        'final_recall': metrics.get('final_recall', 0),
                        'final_precision': metrics.get('final_precision', 0),
                        'confusion_matrix': metrics.get('confusion_matrix', []),
                        'val_precisions': metrics.get('val_precisions', []),
                        'val_f1s': metrics.get('val_f1s', []),
                        'val_recalls': metrics.get('val_recalls', []),
                        'val_accuracies': metrics.get('val_accuracies', []),
                        'train_losses': metrics.get('train_losses', []),
                        'val_losses': metrics.get('val_losses', [])
                    },
                    'epochs': epochs,
                    'batch_size': batch_size
                })
                
                return render_template('train_progress.html')
                
        except Exception as e:
            update_train_progress({
                'status': 'error',
                'log': f'Ошибка обучения: {e}'
            })
            return render_template('train.html', error=f'Ошибка обучения: {e}\n{traceback.format_exc()}', stats=None)
    
    return render_template('train.html', stats=None)

# Endpoint для live-обновления прогресса обучения
from flask import Blueprint
train_bp = Blueprint('train_progress', __name__)

@train_bp.route('/train_progress')
def train_progress():
    global last_train_metrics
    if not last_train_metrics:
        return jsonify({'status': 'not_started'})
    return jsonify(last_train_metrics)

def process_training(xml_files, zip_files, batch_size, epochs):
    # Здесь будет логика запуска обучения модели
    # Возвращать результат, логи, ошибки
    pass

def get_train_logs():
    # Здесь можно реализовать получение логов обучения для live-обновления
    pass 
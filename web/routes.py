import os
import tempfile
import shutil
import zipfile
import json
import logging
from flask import Flask, request, render_template, send_file, redirect, url_for, flash, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import pandas as pd
from datetime import datetime
import uuid

from core.xml_utils import (
    load_all_category_pairs, get_pair, parse_xml_data, export_to_bimstep_xml, export_to_xml, add_bimstep_journal_entry
)
from core.image_utils import find_image_by_name, get_relative_image_path
from ml.model import create_model
from ml.dataset import CollisionImageDataset, create_transforms
from ml.inference import predict, fill_xml_fields
from ml.train import train_model
from web.utils import safe_filename_with_cyrillic, load_settings, save_settings
from web.progress import train_progress, update_train_progress

logger = logging.getLogger(__name__)

# Явно указываем путь к шаблонам
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, 'templates'),
    static_folder=os.path.join(BASE_DIR, 'static')
)
app.secret_key = 'supersecretkey'

UPLOAD_FOLDER = 'temp_uploads'
ALLOWED_EXTENSIONS = {'xml', 'zip'}

# Словарь для хранения соответствий между session_id и путями к временным папкам
session_dirs = {}

def apply_manual_review(df, session_dir):
    """Применяет ручную разметку к DataFrame из файла manual_review.json, обновляя только строки с clash_uid из разметки, остальные не трогает."""
    review_path = os.path.join(session_dir, 'manual_review.json')
    if not os.path.exists(review_path):
        logger.info(f"Файл ручной разметки не найден: {review_path}")
        return df
    try:
        import json
        with open(review_path, 'r', encoding='utf-8') as f:
            reviews = json.load(f)
        logger.info(f"[apply_manual_review] Прочитано {len(reviews)} разметок из {review_path}")
        review_map = {r['clash_uid']: r['status'] for r in reviews}
        df_updated = df.copy()
        # Добавляем колонки для исходных значений, если их нет
        if 'original_prediction_source' not in df_updated.columns:
            df_updated['original_prediction_source'] = df_updated['prediction_source']
        if 'original_cv_prediction' not in df_updated.columns:
            df_updated['original_cv_prediction'] = df_updated['cv_prediction']
        applied_count = 0
        for idx, row in df_updated.iterrows():
            cid = row.get('clash_uid')
            if cid in review_map:
                status = review_map[cid]
                applied_count += 1
                # Сохраняем исходные значения только если это первая ручная разметка
                if pd.isna(row.get('original_prediction_source')):
                    df_updated.at[idx, 'original_prediction_source'] = row.get('prediction_source')
                if pd.isna(row.get('original_cv_prediction')):
                    df_updated.at[idx, 'original_cv_prediction'] = row.get('cv_prediction')
                # Обновляем только нужные поля
                if status == 'Approved':
                    df_updated.at[idx, 'cv_prediction'] = 0
                    df_updated.at[idx, 'cv_confidence'] = 1.0
                    df_updated.at[idx, 'prediction_source'] = 'manual_review'
                    df_updated.at[idx, 'cv_status'] = 'Approved'
                elif status == 'Active':
                    df_updated.at[idx, 'cv_prediction'] = 1
                    df_updated.at[idx, 'cv_confidence'] = 1.0
                    df_updated.at[idx, 'prediction_source'] = 'manual_review'
                    df_updated.at[idx, 'cv_status'] = 'Active'
                else:  # Reviewed
                    df_updated.at[idx, 'cv_prediction'] = -1
                    df_updated.at[idx, 'cv_confidence'] = 0.5
                    df_updated.at[idx, 'prediction_source'] = 'manual_review'
                    df_updated.at[idx, 'cv_status'] = 'Reviewed'
        logger.info(f"Применена ручная разметка к {applied_count} коллизиям из {len(reviews)} разметок")
        # Логируем статистику после применения ручной разметки
        manual_approved = len(df_updated[(df_updated['prediction_source'] == 'manual_review') & (df_updated['cv_prediction'] == 0)])
        manual_active = len(df_updated[(df_updated['prediction_source'] == 'manual_review') & (df_updated['cv_prediction'] == 1)])
        manual_reviewed = len(df_updated[(df_updated['prediction_source'] == 'manual_review') & (df_updated['cv_prediction'] == -1)])
        logger.info(f"Статистика ручной разметки после применения: Approved={manual_approved}, Active={manual_active}, Reviewed={manual_reviewed}")
        return df_updated
    except Exception as e:
        logger.error(f"Ошибка применения ручной разметки: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return df

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def collect_dataset_with_session_dir(xml_path_name_pairs, session_dir, export_format='standard', manual_review_enabled=False):
    """Собирает датасет из XML файлов с учетом временной папки сессии и формата экспорта"""
    all_data = []
    
    for xml_path, orig_filename in xml_path_name_pairs:
        try:
            # Парсим XML файл
            df_file = parse_xml_data(xml_path, export_format=export_format)
            if df_file.empty:
                continue
            # Приведение типов для числовых колонок
            for col in ['cv_prediction', 'cv_confidence', 'clash_x', 'clash_y', 'clash_z', 'distance', 'IsResolved']:
                if col in df_file.columns:
                    df_file[col] = pd.to_numeric(df_file[col], errors='coerce')
            # Добавляем информацию о файле
            df_file['source_file'] = orig_filename
            # Ищем изображения если нужен инференс или ручная разметка
            need_images = (export_format == 'standard') or manual_review_enabled
            if need_images:
                for idx, row in df_file.iterrows():
                    image_href = row.get('image_href', '')
                    rel_path = get_relative_image_path(image_href)
                    image_path = os.path.join(session_dir, rel_path) if rel_path else ''
                    if not image_path or not os.path.exists(image_path):
                        # fallback
                        image_name = os.path.basename(image_href)
                        image_path = find_image_by_name(image_name, session_dir)
                    if image_path and os.path.exists(image_path):
                        df_file.at[idx, 'image_file'] = image_path
                    else:
                        df_file.at[idx, 'image_file'] = ''
            else:
                # Если изображения не нужны, просто добавляем пустую колонку
                df_file['image_file'] = ''
            all_data.append(df_file)
        except Exception as e:
            logger.error(f"Ошибка обработки файла {xml_path}: {e}")
            continue
    
    if all_data:
        df = pd.concat(all_data, ignore_index=True)
        # Добавляем clash_uid, если его нет
        if 'clash_uid' not in df.columns and 'element1_id' in df.columns and 'element2_id' in df.columns:
            def make_uid(row):
                id1 = str(row['element1_id']) if pd.notna(row['element1_id']) else ''
                id2 = str(row['element2_id']) if pd.notna(row['element2_id']) else ''
                return f"{min(id1, id2)}_{max(id1, id2)}" if id1 and id2 else ''
            df['clash_uid'] = df.apply(make_uid, axis=1)
        return df
    else:
        return pd.DataFrame()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/settings', methods=['GET', 'POST'])
def settings():
    if request.method == 'POST':
        action = request.form.get('action', 'save')
        
        if action == 'save':
            settings = load_settings()
            settings['model_file'] = request.form.get('model_file', 'model_clashmark.pt')
            settings['low_confidence'] = float(request.form.get('low_confidence', 0.3))
            settings['high_confidence'] = float(request.form.get('high_confidence', 0.7))
            settings['inference_mode'] = request.form.get('inference_mode', 'model')
            settings['export_format'] = request.form.get('export_format', 'standard')
            settings['manual_review_enabled'] = 'manual_review_enabled' in request.form
            save_settings(settings)
            flash('Настройки сохранены!', 'success')
            
        elif action == 'delete_model':
            model_file = request.form.get('model_file')
            if model_file and model_file != 'model_clashmark.pt':
                model_path = os.path.join('model', model_file)
                stats_path = os.path.join('model', f'{model_file}_stats.json')
                try:
                    if os.path.exists(model_path):
                        os.remove(model_path)
                    if os.path.exists(stats_path):
                        os.remove(stats_path)
                    flash(f'Модель {model_file} удалена!', 'success')
                except Exception as e:
                    flash(f'Ошибка удаления модели: {e}', 'error')
            else:
                flash('Нельзя удалить основную модель!', 'error')
                
        elif action == 'clear_cache':
            try:
                # Удаляем только файлы статистики, не модели
                for file in os.listdir('model'):
                    if file.endswith('_stats.json'):
                        os.remove(os.path.join('model', file))
                flash('Кэш очищен!', 'success')
            except Exception as e:
                flash(f'Ошибка очистки кэша: {e}', 'error')
    
    # Загружаем настройки
    settings = load_settings()
    
    # Получаем список моделей
    model_files = ['model_clashmark.pt']
    if os.path.exists('model'):
        for file in os.listdir('model'):
            if file.endswith('.pt') and file != 'model_clashmark.pt':
                model_files.append(file)
    
    # Загружаем метрики для каждой модели
    model_metrics = {}
    for model_file in model_files:
        stats_file = os.path.join('model', f'{model_file}_stats.json')
        if os.path.exists(stats_file):
            try:
                with open(stats_file, 'r', encoding='utf-8') as f:
                    stats = json.load(f)
                    # Если есть вложенный metrics — разворачиваем его на верхний уровень
                    if 'metrics' in stats and isinstance(stats['metrics'], dict):
                        for k, v in stats['metrics'].items():
                            stats[k] = v
                    model_metrics[model_file] = stats
            except Exception as e:
                logger.error(f"Ошибка загрузки статистики для {model_file}: {e}")
    
    return render_template('settings.html', settings=settings, model_files=model_files, model_metrics=model_metrics)

@app.route('/api/settings', methods=['GET'])
def api_settings():
    settings = load_settings()
    return jsonify(settings)

@app.route('/analyze', methods=['POST'])
def analyze_files():
    try:
        xml_files = request.files.getlist('xml_file')
        zip_files = request.files.getlist('images_zip')
        # Загружаем настройки заранее
        settings = load_settings()
        export_format = settings.get('export_format', 'standard')
        inference_mode = settings.get('inference_mode', 'model')
        low_confidence = settings.get('low_confidence', 0.3)
        high_confidence = settings.get('high_confidence', 0.7)
        manual_review_enabled = settings.get('manual_review_enabled', False)
        analysis_settings = {
            'inference_mode': inference_mode,
            'manual_review_enabled': manual_review_enabled,
            'export_format': export_format,
            'model_file': settings.get('model_file', 'model_clashmark.pt')
        }
        if not xml_files:
            return jsonify({'error': 'Не выбраны XML файлы!', 'analysis_settings': analysis_settings})
        # Для стандартного формата требуем ZIP архивы, для BIM Step - не обязательно
        if export_format == 'standard' and not zip_files:
            return jsonify({'error': 'Для стандартного формата необходимо загрузить ZIP архивы с изображениями!', 'analysis_settings': analysis_settings})
        
        # Создаем временную папку для сессии
        session_dir = tempfile.mkdtemp(prefix='analysis_session_')
        session_id = os.path.basename(session_dir)
        session_dirs[session_id] = session_dir
        
        xml_paths = []
        zip_paths = []
        original_xml_names = []
        xml_path_name_pairs = []
        
        # Сохраняем XML файлы
        for xml_file in xml_files:
            if not xml_file.filename:
                continue
            safe_filename = safe_filename_with_cyrillic(xml_file.filename)
            xml_path = os.path.join(session_dir, safe_filename)
            xml_file.save(xml_path)
            xml_paths.append(xml_path)
            original_xml_names.append(os.path.splitext(safe_filename)[0])
            xml_path_name_pairs.append((xml_path, safe_filename))
        
        # Обрабатываем ZIP файлы только если они загружены
        if zip_files:
            for zip_file in zip_files:
                if not zip_file.filename:
                    continue
                zip_path = os.path.join(session_dir, safe_filename_with_cyrillic(zip_file.filename))
                zip_file.save(zip_path)
                zip_paths.append(zip_path)
            
            # Распаковываем ZIP файлы
            for zip_path in zip_paths:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(session_dir)
        
        # Собираем датасет
        df = collect_dataset_with_session_dir(xml_path_name_pairs, session_dir, export_format=export_format, manual_review_enabled=manual_review_enabled)
        
        if df.empty:
            return jsonify({'error': 'Не удалось обработать XML файлы!', 'analysis_settings': analysis_settings})
        
        # Загружаем категории пар
        pairs = load_all_category_pairs('category_pairs.yaml')
        can_pairs = set()
        cannot_pairs = set()
        visual_pairs = set()
        
        if 'can' in pairs and isinstance(pairs['can'], list):
            for pair in pairs['can']:
                if isinstance(pair, list) and len(pair) == 2:
                    a, b = pair
                    can_pairs.add((a, b) if a <= b else (b, a))
        
        if 'cannot' in pairs and isinstance(pairs['cannot'], list):
            for pair in pairs['cannot']:
                if isinstance(pair, list) and len(pair) == 2:
                    a, b = pair
                    cannot_pairs.add((a, b) if a <= b else (b, a))
        
        if 'visual' in pairs and isinstance(pairs['visual'], list):
            for pair in pairs['visual']:
                if isinstance(pair, list) and len(pair) == 2:
                    a, b = pair
                    visual_pairs.add((a, b) if a <= b else (b, a))
        
        # Алгоритм распределяет коллизии
        df['cv_prediction'] = None
        df['cv_confidence'] = None
        df['prediction_source'] = None
        df['cv_status'] = None # Добавляем колонку для статуса
        
        visual_rows = []
        visual_indices = []
        
        for idx, row in df.iterrows():
            pair = get_pair(row)
            if pair in can_pairs:
                # can -> Approved
                df.at[idx, 'cv_prediction'] = 0  # Approved
                df.at[idx, 'cv_confidence'] = 1.0
                df.at[idx, 'prediction_source'] = 'algorithm'
                df.at[idx, 'cv_status'] = 'Approved'
            elif pair in cannot_pairs:
                # cannot -> Active
                df.at[idx, 'cv_prediction'] = 1  # Active
                df.at[idx, 'cv_confidence'] = 1.0
                df.at[idx, 'prediction_source'] = 'algorithm'
                df.at[idx, 'cv_status'] = 'Active'
            elif pair in visual_pairs:
                # visual - требует дополнительной обработки
                df.at[idx, 'cv_prediction'] = -1  # visual (временно)
                df.at[idx, 'cv_confidence'] = 0.5
                df.at[idx, 'prediction_source'] = 'algorithm'
                df.at[idx, 'cv_status'] = 'Reviewed' # По умолчанию для visual
                visual_rows.append(row)
                visual_indices.append(idx)
            else:
                # Неизвестная пара - считаем visual
                df.at[idx, 'cv_prediction'] = -1  # visual
                df.at[idx, 'cv_confidence'] = 0.5
                df.at[idx, 'prediction_source'] = 'algorithm'
                df.at[idx, 'cv_status'] = 'Reviewed' # По умолчанию для visual
                visual_rows.append(row)
                visual_indices.append(idx)
        
        # Обрабатываем visual коллизии
        if visual_rows:
            if inference_mode == 'model':
                # Используем модель для разметки visual коллизий
                import torch
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                visual_df = pd.DataFrame(visual_rows)
                transform = create_transforms(is_training=False)
                model = create_model(device)
                model_path = os.path.join('model', settings.get('model_file', 'model_clashmark.pt'))
                model.load_state_dict(torch.load(model_path, map_location=device))
                model.to(device)
                model.eval()
                # Используем пороги уверенности из настроек
                visual_pred_df = predict(model, device, visual_df, transform, 
                                       low_confidence_threshold=low_confidence, 
                                       high_confidence_threshold=high_confidence,
                                       session_dir=session_dir)
                for i, idx in enumerate(visual_indices):
                    prediction = int(visual_pred_df.iloc[i]['cv_prediction'])
                    confidence = float(visual_pred_df.iloc[i]['cv_confidence'])
                    if prediction == -1:
                        # Модель сомневается - явно присваиваем Reviewed
                        df.at[idx, 'cv_prediction'] = -1  # visual -> Reviewed
                        df.at[idx, 'cv_confidence'] = confidence
                        df.at[idx, 'prediction_source'] = 'model_uncertain'
                        df.at[idx, 'cv_status'] = 'Reviewed'
                    else:
                        # Модель уверена
                        df.at[idx, 'cv_prediction'] = prediction  # 0 -> Approved, 1 -> Active
                        df.at[idx, 'cv_confidence'] = confidence
                        df.at[idx, 'prediction_source'] = 'model'
                        df.at[idx, 'cv_status'] = 'Approved' if prediction == 0 else 'Active'
            elif manual_review_enabled:
                # Режим ручной разметки - оставляем visual для ручной обработки
                for idx in visual_indices:
                    df.at[idx, 'cv_prediction'] = -1  # visual -> Reviewed (для ручной разметки)
                    df.at[idx, 'cv_confidence'] = 0.5
                    df.at[idx, 'prediction_source'] = 'manual_review'
                    df.at[idx, 'cv_status'] = 'Reviewed'
            else:
                # Ни модель, ни ручная разметка не используются - все visual -> Reviewed
                for idx in visual_indices:
                    df.at[idx, 'cv_prediction'] = -1  # visual -> Reviewed
                    df.at[idx, 'cv_confidence'] = 0.5
                    df.at[idx, 'prediction_source'] = 'algorithm'
                    df.at[idx, 'cv_status'] = 'Reviewed'
        
        # Применяем ручную разметку к общему DataFrame для корректного подсчета статистики и экспорта
        df_with_manual = apply_manual_review(df, session_dir)
        
        # --- Все дальнейшие действия (экспорт, статистика) только по df_with_manual ---
        # Логируем статистику после применения ручной разметки
        total_approved = int((df_with_manual['cv_prediction'] == 0).sum())
        total_active = int((df_with_manual['cv_prediction'] == 1).sum())
        total_reviewed = int((df_with_manual['cv_prediction'] == -1).sum())
        
        logger.info(f"Итоговая статистика после ручной разметки: Approved={total_approved}, Active={total_active}, Reviewed={total_reviewed}")
        
        # Логируем распределение по источникам разметки
        manual_count = len(df_with_manual[df_with_manual['prediction_source'] == 'manual_review'])
        algorithm_count = len(df_with_manual[df_with_manual['prediction_source'] == 'algorithm'])
        model_count = len(df_with_manual[df_with_manual['prediction_source'] == 'model'])
        model_uncertain_count = len(df_with_manual[df_with_manual['prediction_source'] == 'model_uncertain'])
        
        logger.info(f"Распределение по источникам: Manual={manual_count}, Algorithm={algorithm_count}, Model={model_count}, Model_Uncertain={model_uncertain_count}")
        
        # Экспорт и статистика только по df_with_manual
        download_links = []
        stats_per_file = []
        
        for i, (xml_path, orig_filename) in enumerate(xml_path_name_pairs):
            orig_base_name = os.path.splitext(orig_filename)[0]
            
            if export_format == 'bimstep':
                output_xml = os.path.join(session_dir, f"bimstep_results_{orig_base_name}.xml")
            else:
                output_xml = os.path.join(session_dir, f"cv_results_{orig_base_name}.xml")
            
            # Фильтруем данные по файлу из DataFrame с примененной ручной разметкой
            df_file = df_with_manual[df_with_manual['source_file'] == orig_filename].copy()
            
            if not df_file.empty:
                # Логируем статистику перед экспортом
                approved_count = (df_file['cv_prediction'] == 0).sum()
                active_count = (df_file['cv_prediction'] == 1).sum()
                reviewed_count = (df_file['cv_prediction'] == -1).sum()
                logger.info(f"Экспорт {orig_filename}: Approved={approved_count}, Active={active_count}, Reviewed={reviewed_count}")
                
                # Логируем распределение по источникам разметки для этого файла
                manual_count = len(df_file[df_file['prediction_source'] == 'manual_review'])
                algorithm_count = len(df_file[df_file['prediction_source'] == 'algorithm'])
                model_count = len(df_file[df_file['prediction_source'] == 'model'])
                model_uncertain_count = len(df_file[df_file['prediction_source'] == 'model_uncertain'])
                logger.info(f"Источники разметки для {orig_filename}: Manual={manual_count}, Algorithm={algorithm_count}, Model={model_count}, Model_Uncertain={model_uncertain_count}")
                
                # Экспортируем результаты с учетом ручной разметки
                if export_format == 'bimstep':
                    export_to_bimstep_xml(df_file, output_xml, xml_path)
                    # Добавляем записи в журнал BIM Step
                    for _, row in df_file.iterrows():
                        if row.get('cv_prediction') is not None:
                            prediction_source = row.get('prediction_source', '')
                            if prediction_source == 'model':
                                prediction_type = 'model'
                            elif prediction_source == 'manual_review':
                                prediction_type = 'manual'
                            else:
                                prediction_type = 'algorithm'
                            
                            # Определяем статус для журнала
                            if row.get('cv_prediction') == 0:
                                status = 'approved'
                            elif row.get('cv_prediction') == 1:
                                status = 'active'
                            else:
                                status = 'reviewed'
                            
                            add_bimstep_journal_entry(
                                row.get('clash_uid', ''), 
                                prediction_type, 
                                'Разметка с помощью ClashMark', 
                                session_dir=session_dir,
                                element1_id=row.get('element1_id', ''),
                                element2_id=row.get('element2_id', ''),
                                status=status
                            )
                else:
                    export_to_xml(df_file, output_xml, xml_path)
                
                # Создаем ссылки для скачивания
                download_url = url_for('download_file', session_id=session_id, filename=os.path.basename(output_xml))
                download_links.append({'name': os.path.basename(output_xml), 'url': download_url})
                
                # Добавляем JournalBimStep.xml для BIM Step формата
                if export_format == 'bimstep':
                    journal_path = os.path.join(session_dir, 'JournalBimStep.xml')
                    if os.path.exists(journal_path):
                        journal_download_url = url_for('download_file', session_id=session_id, filename='JournalBimStep.xml')
                        download_links.append({'name': 'JournalBimStep.xml', 'url': journal_download_url})
                
                # Итоговая статистика по файлу
                total_collisions = len(df_file)
                found_images = pd.Series(df_file['image_file']).notna().sum() if 'image_file' in df_file.columns else 0
                approved_count = (df_file['cv_prediction'] == 0).sum()  # can -> Approved
                active_count = (df_file['cv_prediction'] == 1).sum()    # cannot -> Active
                reviewed_count = (df_file['cv_prediction'] == -1).sum()  # visual -> Reviewed
                stats_per_file.append({
                    'file': f'{orig_base_name}.xml',
                    'total_collisions': total_collisions,
                    'found_images': found_images,
                    'approved_count': approved_count,
                    'active_count': active_count,
                    'reviewed_count': reviewed_count
                })
        # После формирования stats_per_file, добавляем stats_total для фронта
        # Пересчитываем итоговую статистику с учетом ручной разметки
        total_approved = (df_with_manual['cv_prediction'] == 0).sum()
        total_active = (df_with_manual['cv_prediction'] == 1).sum()
        total_reviewed = (df_with_manual['cv_prediction'] == -1).sum()
        
        stats_total = {
            'total_files': len(stats_per_file),
            'total_collisions': len(df_with_manual),
            'total_approved': total_approved,
            'total_active': total_active,
            'total_reviewed': total_reviewed
        }
        
        # Логируем итоговую статистику
        logger.info(f"Итоговая статистика для фронта: {stats_total}")
        
        # --- Детальная статистика по типам разметки ---
        detailed_stats = []
        for i, (xml_path, orig_filename) in enumerate(xml_path_name_pairs):
            orig_base_name = os.path.splitext(orig_filename)[0]
            
            # Фильтруем данные по файлу из DataFrame с примененной ручной разметкой
            df_file = df_with_manual[df_with_manual['source_file'] == orig_filename].copy()
            
            if not df_file.empty:
                # Корректно определяем источник и значение предсказания с учётом ручной разметки
                pred_source = df_file['original_prediction_source'].combine_first(df_file['prediction_source']) if 'original_prediction_source' in df_file.columns else df_file['prediction_source']
                pred_value = df_file['original_cv_prediction'].combine_first(df_file['cv_prediction']) if 'original_cv_prediction' in df_file.columns else df_file['cv_prediction']
                # Статистика по алгоритму
                algorithm_approved = int(((pred_source == 'algorithm') & (pred_value == 0)).sum())
                algorithm_active = int(((pred_source == 'algorithm') & (pred_value == 1)).sum())
                algorithm_reviewed = int(((pred_source == 'algorithm') & (pred_value == -1)).sum())
                # Статистика по модели
                model_mask = pred_source.isin(['model', 'model_uncertain'])
                model_approved = int(((model_mask) & (pred_value == 0)).sum())
                model_active = int(((model_mask) & (pred_value == 1)).sum())
                model_reviewed = int(((model_mask) & (pred_value == -1)).sum())
                # Статистика по ручной разметке
                manual_approved = int(((df_file['prediction_source'] == 'manual_review') & (df_file['cv_prediction'] == 0)).sum())
                manual_active = int(((df_file['prediction_source'] == 'manual_review') & (df_file['cv_prediction'] == 1)).sum())
                manual_reviewed = int(((df_file['prediction_source'] == 'manual_review') & (df_file['cv_prediction'] == -1)).sum())
                
                # value_counts по статусу
                status_counts = df_file['cv_status'].value_counts().to_dict()
                
                detailed_stats.append({
                    'file_name': orig_filename,
                    'total_collisions': len(df_file),
                    'algorithm': {
                        'approved': algorithm_approved,
                        'active': algorithm_active,
                        'reviewed': algorithm_reviewed
                    },
                    'model': {
                        'approved': model_approved,
                        'active': model_active,
                        'reviewed': model_reviewed
                    },
                    'manual': {
                        'approved': manual_approved,
                        'active': manual_active,
                        'reviewed': manual_reviewed
                    },
                    'status_counts': status_counts
                })
                
                # Логируем детальную статистику для этого файла
                logger.info(f"Детальная статистика для {orig_filename}: Algorithm(A={algorithm_approved},Ac={algorithm_active},R={algorithm_reviewed}), Model(A={model_approved},Ac={model_active},R={model_reviewed}), Manual(A={manual_approved},Ac={manual_active},R={manual_reviewed})")
        
        # --- Новый блок: формируем список для ручной разметки ---
        manual_review_collisions = []
        if manual_review_enabled:
            # Используем исходный DataFrame для формирования списка ручной разметки
            # Нужно найти коллизии с cv_prediction == -1 (которые требуют ручной разметки)
            for _, row in df.iterrows():
                if row.get('cv_prediction') == -1:
                    image_path = row.get('image_file', '')
                    # Получаем относительный путь от session_dir, если image_path абсолютный
                    if image_path and image_path.startswith(session_dir):
                        rel_image_path = os.path.relpath(image_path, session_dir)
                    elif image_path:
                        rel_image_path = image_path
                    else:
                        rel_image_path = ''
                    # Логгируем для отладки
                    logger.info(f"MANUAL_REVIEW: clash_uid={row.get('clash_uid', '')}, image_file={rel_image_path}")
                    manual_review_collisions.append({
                        'clash_uid': row.get('clash_uid', ''),
                        'image_file': rel_image_path,
                        'element1_category': row.get('element1_category', ''),
                        'element2_category': row.get('element2_category', ''),
                        'description': row.get('clash_name', ''),
                        'source_file': row.get('source_file', ''),
                        'session_id': session_id
                    })
            logger.info(f"Сформирован список для ручной разметки: {len(manual_review_collisions)} коллизий")
        else:
            logger.info("Ручная разметка отключена в настройках")
        
        response_data = {
            'success': True,
            'session_id': session_id,
            'download_links': download_links,
            'stats_per_file': stats_per_file,
            'stats_total': stats_total,
            'used_images': export_format == 'standard',
            'analysis_settings': analysis_settings,
            'manual_review_collisions': manual_review_collisions,
            'detailed_stats': detailed_stats
        }
        
        logger.info(f"Отправляем ответ с {len(download_links)} ссылками для скачивания и {len(detailed_stats)} детальными статистиками")
        logger.info(f"Итоговая статистика в ответе: {stats_total}")
        # --- В analyze_files (после формирования df_with_manual и перед return) ---
        csv_path = os.path.join(session_dir, 'df_with_inference.csv')
        df_with_manual.to_csv(csv_path, index=False)
        return jsonify(to_py(response_data))
        
    except Exception as e:
        logger.error(f"Ошибка анализа файлов: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({'error': f'Ошибка анализа: {str(e)}'})

@app.route('/api/analyze_preview', methods=['POST'])
def analyze_preview():
    """Анализирует загруженные XML и ZIP-файлы, возвращает статистику для предпросмотра (анализ коллизий)"""
    try:
        xml_files = request.files.getlist('xml_file')
        zip_files = request.files.getlist('images_zip')
        if not xml_files:
            return jsonify({'error': 'Не выбраны XML-файлы'}), 400
        with tempfile.TemporaryDirectory() as session_dir:
            xml_paths = []
            for xml_file in xml_files:
                if not xml_file.filename:
                    continue
                safe_filename = safe_filename_with_cyrillic(xml_file.filename)
                xml_path = os.path.join(session_dir, safe_filename)
                xml_file.save(xml_path)
                xml_paths.append(xml_path)
            zip_paths = []
            for zip_file in zip_files:
                if not zip_file.filename:
                    continue
                zip_path = os.path.join(session_dir, safe_filename_with_cyrillic(zip_file.filename))
                zip_file.save(zip_path)
                zip_paths.append(zip_path)
            for zip_path in zip_paths:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(session_dir)
            # Загружаем настройки для определения формата экспорта
            settings = load_settings()
            export_format = settings.get('export_format', 'standard')
            from collections import Counter
            all_dfs = []
            for xml_path in xml_paths:
                try:
                    df = parse_xml_data(xml_path, export_format)
                    all_dfs.append(df)
                except Exception as e:
                    continue
            if not all_dfs:
                return jsonify({'error': 'Не удалось обработать XML-файлы'}), 400
            df = pd.concat(all_dfs, ignore_index=True)
            pair_counts = Counter([get_pair(row) for _, row in df.iterrows()])
            pairs_with_counts = sorted([[str(p[0]), str(p[1]), int(c)] for p, c in pair_counts.items()], key=lambda x: (-int(x[2] or 0), str(x[0]), str(x[1])))
            if 'image_href' in df.columns:
                image_files = pd.Series(df['image_href']).dropna().unique()
                image_count = len(image_files)
            else:
                image_count = 0
            stats = {
                'xml_file_count': len(xml_files),
                'zip_file_count': len(zip_files),
                'total_collisions': len(df),
                'image_count': image_count,
                'category_pairs': pairs_with_counts
            }
            return jsonify(to_py({'success': True, 'stats': stats}))
    except Exception as e:
        return jsonify({'error': f'Ошибка анализа файлов: {e}'}), 500

@app.route('/api/train_preview', methods=['POST'])
def train_preview():
    """Анализирует загруженные XML и ZIP-файлы, возвращает статистику для предпросмотра (обучение модели)"""
    try:
        xml_files = request.files.getlist('xml_file')
        zip_files = request.files.getlist('zip_file')
        if not xml_files:
            return jsonify({'error': 'Не выбраны XML-файлы'}), 400
        with tempfile.TemporaryDirectory() as session_dir:
            xml_paths = []
            xml_file_names = []
            for xml_file in xml_files:
                if not xml_file.filename:
                    continue
                safe_filename = safe_filename_with_cyrillic(xml_file.filename)
                xml_path = os.path.join(session_dir, safe_filename)
                xml_file.save(xml_path)
                xml_paths.append(xml_path)
                xml_file_names.append(safe_filename)
            zip_paths = []
            for zip_file in zip_files:
                if not zip_file.filename:
                    continue
                zip_path = os.path.join(session_dir, safe_filename_with_cyrillic(zip_file.filename))
                zip_file.save(zip_path)
                zip_paths.append(zip_path)
            for zip_path in zip_paths:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(session_dir)
            # Загружаем настройки для определения формата экспорта
            settings = load_settings()
            export_format = settings.get('export_format', 'standard')
            from collections import Counter
            all_dfs = []
            for xml_path in xml_paths:
                try:
                    df = parse_xml_data(xml_path, export_format)
                    all_dfs.append(df)
                except Exception as e:
                    continue
            if not all_dfs:
                return jsonify({'error': 'Не удалось обработать XML-файлы'}), 400
            df = pd.concat(all_dfs, ignore_index=True)
            df_train = df[df['IsResolved'] != -1]
            # Только visual-пары
            import yaml
            with open('category_pairs.yaml', 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            visual_pairs = set()
            if 'visual' in data and isinstance(data['visual'], list):
                for pair in data['visual']:
                    if isinstance(pair, list) and len(pair) == 2:
                        a, b = pair
                        visual_pairs.add((a, b) if a <= b else (b, a))
            pair_counts = Counter([get_pair(row) for _, row in df_train.iterrows() if get_pair(row) in visual_pairs])
            pairs_with_counts = sorted([[str(p[0]), str(p[1]), int(c)] for p, c in pair_counts.items()], key=lambda x: (-int(x[2] or 0), str(x[0]), str(x[1])))
            if 'image_href' in df_train.columns:
                image_files = pd.Series(df_train['image_href']).dropna().unique()
                image_count = len(image_files)
            else:
                image_count = 0
            trainable_collisions = sum([int(x[2] or 0) for x in pairs_with_counts])
            # Подробная статистика по каждому XML-файлу
            per_file = []
            for xml_path, xml_name in zip(xml_paths, xml_file_names):
                try:
                    df_file = parse_xml_data(xml_path, export_format)
                except Exception as e:
                    continue
                total_collisions = len(df_file)
                active_count = int((df_file['IsResolved'] == 1).sum()) if 'IsResolved' in df_file.columns else 0
                approve_count = int((df_file['IsResolved'] == 0).sum()) if 'IsResolved' in df_file.columns else 0
                reviewed_count = int((df_file['IsResolved'] == -1).sum()) if 'IsResolved' in df_file.columns else 0
                per_file.append({
                    'file': xml_name,
                    'total_collisions': total_collisions,
                    'active_count': active_count,
                    'approve_count': approve_count,
                    'reviewed_count': reviewed_count
                })
            stats = {
                'xml_file_count': len(xml_files),
                'zip_file_count': len(zip_files),
                'total_collisions': len(df),
                'trainable_collisions': trainable_collisions,
                'category_pairs': pairs_with_counts,
                'per_file': per_file
            }
            logging.debug(f"train_preview stats: {stats}")
            try:
                debug_per_file_types = [str(type(f)) + ':' + str(f) for f in per_file]
            except Exception as e:
                debug_per_file_types = str(e)
            stats['debug_per_file_types'] = debug_per_file_types
            return jsonify(to_py({'success': True, 'stats': stats}))
    except Exception as e:
        return jsonify({'error': f'Ошибка анализа файлов: {e}'}), 500

@app.route('/api/manual_review', methods=['POST'])
def api_manual_review():
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        reviews = data.get('reviews', [])
        logger.info(f"[manual_review] session_id={session_id}, reviews_count={len(reviews)}")
        if not session_id or not reviews:
            logger.error(f"[manual_review] Нет session_id или разметок: session_id={session_id}, reviews={reviews}")
            return jsonify({'error': 'Нет session_id или разметок'}), 400
        session_dir = session_dirs.get(session_id)
        if not session_dir or not os.path.exists(session_dir):
            logger.error(f"[manual_review] Сессия не найдена: {session_id}")
            return jsonify({'error': 'Сессия не найдена'}), 404
        # Проверяем каждую разметку
        for i, review in enumerate(reviews):
            if 'clash_uid' not in review or 'status' not in review:
                logger.error(f"[manual_review] Некорректная разметка в позиции {i}: {review}")
                return jsonify({'error': f'Некорректная разметка в позиции {i}: {review}'}), 400
        # Логируем первые 5 для отладки
        for review in reviews[:5]:
            logger.info(f"  clash_uid: {review.get('clash_uid')}, status: {review.get('status')}")
        # Подсчитываем статистику разметки
        status_counts = {}
        for review in reviews:
            status = review.get('status', 'Unknown')
            status_counts[status] = status_counts.get(status, 0) + 1
        logger.info(f"Статистика ручной разметки: {status_counts}")
        # Сохраняем разметку в файл
        review_path = os.path.join(session_dir, 'manual_review.json')
        import json
        try:
            with open(review_path, 'w', encoding='utf-8') as f:
                json.dump(reviews, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"[manual_review] Ошибка записи manual_review.json: {e}")
            return jsonify({'error': f'Ошибка записи manual_review.json: {e}'}), 500
        logger.info(f"Ручная разметка сохранена в {review_path}")
        logger.info(f"Статистика сохраненной разметки: {status_counts}")
        # --- Новый блок: применяем ручную разметку, пересчитываем статистику и переэкспортируем XML ---
        import pandas as pd
        df_path = os.path.join(session_dir, 'df_with_inference.csv')
        if os.path.exists(df_path):
            df_combined = pd.read_csv(df_path)
            df_with_manual = apply_manual_review(df_combined, session_dir)
            # value_counts по статусу
            status_counts = df_with_manual['cv_status'].value_counts().to_dict()
            # Переэкспорт XML
            orig_xmls = []
            for f in os.listdir(session_dir):
                if f.endswith('.xml') and not f.startswith(('cv_results_', 'bimstep_results_', 'JournalBimStep')):
                    orig_xmls.append(os.path.join(session_dir, f))
            settings = load_settings()
            export_format = settings.get('export_format', 'standard')
            download_links = []
            for orig_xml in orig_xmls:
                orig_filename = os.path.basename(orig_xml)
                orig_base_name = os.path.splitext(orig_filename)[0]
                df_file = df_with_manual[df_with_manual['source_file'] == orig_filename].copy()
                if not df_file.empty:
                    if export_format == 'bimstep':
                        output_xml = os.path.join(session_dir, f"bimstep_results_{orig_base_name}.xml")
                        export_to_bimstep_xml(df_file, output_xml, orig_xml)
                    else:
                        output_xml = os.path.join(session_dir, f"cv_results_{orig_base_name}.xml")
                        export_to_xml(df_file, output_xml, orig_xml)
                    download_url = url_for('download_file', session_id=session_id, filename=os.path.basename(output_xml))
                    download_links.append({'name': os.path.basename(output_xml), 'url': download_url})
                    if export_format == 'bimstep':
                        journal_path = os.path.join(session_dir, 'JournalBimStep.xml')
                        if os.path.exists(journal_path):
                            journal_download_url = url_for('download_file', session_id=session_id, filename='JournalBimStep.xml')
                            download_links.append({'name': 'JournalBimStep.xml', 'url': journal_download_url})
            # detailed_stats
            detailed_stats = []
            for orig_xml in orig_xmls:
                orig_filename = os.path.basename(orig_xml)
                df_file = df_with_manual[df_with_manual['source_file'] == orig_filename].copy()
                if not df_file.empty:
                    pred_source = df_file['original_prediction_source'].combine_first(df_file['prediction_source']) if 'original_prediction_source' in df_file.columns else df_file['prediction_source']
                    pred_value = df_file['original_cv_prediction'].combine_first(df_file['cv_prediction']) if 'original_cv_prediction' in df_file.columns else df_file['cv_prediction']
                    algorithm_approved = int(((pred_source == 'algorithm') & (pred_value == 0)).sum())
                    algorithm_active = int(((pred_source == 'algorithm') & (pred_value == 1)).sum())
                    algorithm_reviewed = int(((pred_source == 'algorithm') & (pred_value == -1)).sum())
                    model_mask = pred_source.isin(['model', 'model_uncertain'])
                    model_approved = int(((model_mask) & (pred_value == 0)).sum())
                    model_active = int(((model_mask) & (pred_value == 1)).sum())
                    model_reviewed = int(((model_mask) & (pred_value == -1)).sum())
                    manual_approved = int(((df_file['prediction_source'] == 'manual_review') & (df_file['cv_prediction'] == 0)).sum())
                    manual_active = int(((df_file['prediction_source'] == 'manual_review') & (df_file['cv_prediction'] == 1)).sum())
                    manual_reviewed = int(((df_file['prediction_source'] == 'manual_review') & (df_file['cv_prediction'] == -1)).sum())
                    status_counts_file = df_file['cv_status'].value_counts().to_dict()
                    detailed_stats.append({
                        'file_name': orig_filename,
                        'total_collisions': len(df_file),
                        'algorithm': {
                            'approved': algorithm_approved,
                            'active': algorithm_active,
                            'reviewed': algorithm_reviewed
                        },
                        'model': {
                            'approved': model_approved,
                            'active': model_active,
                            'reviewed': model_reviewed
                        },
                        'manual': {
                            'approved': manual_approved,
                            'active': manual_active,
                            'reviewed': manual_reviewed
                        },
                        'status_counts': status_counts_file
                    })
            # Сохраняем актуальный DataFrame
            df_with_manual.to_csv(df_path, index=False)
            return jsonify({'success': True, 'status_counts': status_counts, 'download_links': download_links, 'detailed_stats': detailed_stats})
        return jsonify({'success': True})
    except Exception as e:
        logger.error(f"Ошибка сохранения ручной разметки: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/updated_stats/<session_id>', methods=['GET'])
def api_updated_stats(session_id):
    """Возвращает обновленную статистику после ручной разметки"""
    try:
        logger.info(f"Запрос обновленной статистики для сессии {session_id}")
        session_dir = session_dirs.get(session_id)
        if not session_dir or not os.path.exists(session_dir):
            logger.error(f"Сессия {session_id} не найдена или папка не существует")
            return jsonify({'error': 'Сессия не найдена'}), 404
        # --- Новый способ: загружаем DataFrame с инференсом ---
        import pandas as pd
        df_path = os.path.join(session_dir, 'df_with_inference.csv')
        if not os.path.exists(df_path):
            logger.error(f"Файл с инференсом не найден: {df_path}")
            return jsonify({'error': 'Нет данных для обновления статистики. Проведите анализ заново.'}), 400
        df_combined = pd.read_csv(df_path)
        # Применяем ручную разметку
        df_with_manual = apply_manual_review(df_combined, session_dir)
        
        # Находим исходные XML файлы для detailed_stats и переэкспорта
        orig_xmls = []
        for f in os.listdir(session_dir):
            if f.endswith('.xml') and not f.startswith(('cv_results_', 'bimstep_results_', 'JournalBimStep')):
                orig_xmls.append(os.path.join(session_dir, f))
        
        # Подсчитываем обновленную статистику
        total_approved = int((df_with_manual['cv_prediction'] == 0).sum())
        total_active = int((df_with_manual['cv_prediction'] == 1).sum())
        total_reviewed = int((df_with_manual['cv_prediction'] == -1).sum())
        
        # Логируем статистику для отладки
        logger.info(f"Обновленная статистика после ручной разметки: Approved={total_approved}, Active={total_active}, Reviewed={total_reviewed}")
        
        # Логируем распределение по источникам разметки
        manual_count = len(df_with_manual[df_with_manual['prediction_source'] == 'manual_review'])
        algorithm_count = len(df_with_manual[df_with_manual['prediction_source'] == 'algorithm'])
        model_count = len(df_with_manual[df_with_manual['prediction_source'] == 'model'])
        model_uncertain_count = len(df_with_manual[df_with_manual['prediction_source'] == 'model_uncertain'])
        
        logger.info(f"Распределение по источникам: Manual={manual_count}, Algorithm={algorithm_count}, Model={model_count}, Model_Uncertain={model_uncertain_count}")
        
        # Также возвращаем обновленную детальную статистику
        detailed_stats = []
        for orig_xml in orig_xmls:
            orig_filename = os.path.basename(orig_xml)
            
            # Фильтруем данные по файлу из DataFrame с примененной ручной разметкой
            df_file = df_with_manual[df_with_manual['source_file'] == orig_filename].copy()
            
            if not df_file.empty:
                # Корректно определяем источник и значение предсказания с учётом ручной разметки
                pred_source = df_file['original_prediction_source'].combine_first(df_file['prediction_source']) if 'original_prediction_source' in df_file.columns else df_file['prediction_source']
                pred_value = df_file['original_cv_prediction'].combine_first(df_file['cv_prediction']) if 'original_cv_prediction' in df_file.columns else df_file['cv_prediction']
                # Статистика по алгоритму
                algorithm_approved = int(((pred_source == 'algorithm') & (pred_value == 0)).sum())
                algorithm_active = int(((pred_source == 'algorithm') & (pred_value == 1)).sum())
                algorithm_reviewed = int(((pred_source == 'algorithm') & (pred_value == -1)).sum())
                # Статистика по модели
                model_mask = pred_source.isin(['model', 'model_uncertain'])
                model_approved = int(((model_mask) & (pred_value == 0)).sum())
                model_active = int(((model_mask) & (pred_value == 1)).sum())
                model_reviewed = int(((model_mask) & (pred_value == -1)).sum())
                # Статистика по ручной разметке
                manual_approved = int(((df_file['prediction_source'] == 'manual_review') & (df_file['cv_prediction'] == 0)).sum())
                manual_active = int(((df_file['prediction_source'] == 'manual_review') & (df_file['cv_prediction'] == 1)).sum())
                manual_reviewed = int(((df_file['prediction_source'] == 'manual_review') & (df_file['cv_prediction'] == -1)).sum())
                
                # value_counts по статусу
                status_counts = df_file['cv_status'].value_counts().to_dict()
                
                detailed_stats.append({
                    'file_name': orig_filename,
                    'total_collisions': int(len(df_file)),
                    'algorithm': {
                        'approved': algorithm_approved,
                        'active': algorithm_active,
                        'reviewed': algorithm_reviewed
                    },
                    'model': {
                        'approved': model_approved,
                        'active': model_active,
                        'reviewed': model_reviewed
                    },
                    'manual': {
                        'approved': manual_approved,
                        'active': manual_active,
                        'reviewed': manual_reviewed
                    },
                    'status_counts': status_counts
                })
        
        # Создаем полный ответ с детальной статистикой
        full_response = {
            'total_collisions': int(len(df_with_manual)),
            'total_approved': total_approved,
            'total_active': total_active,
            'total_reviewed': total_reviewed,
            'detailed_stats': detailed_stats
        }
        
        # --- Новый блок: Переэкспорт файлов с учетом ручной разметки ---
        settings = load_settings()
        export_format = settings.get('export_format', 'standard')
        download_links = []  # <--- добавлено
        for orig_xml in orig_xmls:
            orig_filename = os.path.basename(orig_xml)
            orig_base_name = os.path.splitext(orig_filename)[0]
            df_file = df_with_manual[df_with_manual['source_file'] == orig_filename].copy()
            if not df_file.empty:
                if export_format == 'bimstep':
                    output_xml = os.path.join(session_dir, f"bimstep_results_{orig_base_name}.xml")
                    export_to_bimstep_xml(df_file, output_xml, orig_xml)
                else:
                    output_xml = os.path.join(session_dir, f"cv_results_{orig_base_name}.xml")
                    export_to_xml(df_file, output_xml, orig_xml)
                logger.info(f"Переэкспортирован файл {orig_filename} с учетом ручной разметки")
                # --- формируем ссылку для скачивания ---
                download_url = url_for('download_file', session_id=session_id, filename=os.path.basename(output_xml))
                download_links.append({'name': os.path.basename(output_xml), 'url': download_url})
                # Для BIM Step добавляем JournalBimStep.xml
                if export_format == 'bimstep':
                    journal_path = os.path.join(session_dir, 'JournalBimStep.xml')
                    if os.path.exists(journal_path):
                        journal_download_url = url_for('download_file', session_id=session_id, filename='JournalBimStep.xml')
                        download_links.append({'name': 'JournalBimStep.xml', 'url': journal_download_url})
        # --- Конец блока ---
        # Перезаписываем df_with_inference.csv актуальным DataFrame с ручной разметкой
        df_with_manual.to_csv(os.path.join(session_dir, 'df_with_inference.csv'), index=False)
        
        logger.info(f"Возвращаем обновленную статистику: {full_response}")
        logger.info(f"Детальная статистика содержит {len(detailed_stats)} файлов")
        # --- добавляем download_links в ответ ---
        full_response['download_links'] = download_links
        return jsonify(full_response)
    except Exception as e:
        logger.error(f"Ошибка в api_updated_stats: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({'error': f'Ошибка обновления статистики: {str(e)}'}), 500

@app.route('/download/<session_id>/<path:filename>')
def download_file(session_id, filename):
    try:
        logger.debug(f"DOWNLOAD: session_id={session_id}, filename={filename}")
        session_dir = session_dirs.get(session_id)
        logger.debug(f"DOWNLOAD: session_dir={session_dir}")
        if not session_dir or not os.path.exists(session_dir):
            logger.debug(f"DOWNLOAD: session_dir not found or does not exist")
            return "Файл не найден", 404
        file_path = os.path.join(session_dir, filename)
        logger.debug(f"DOWNLOAD: file_path={file_path}")
        
        if not os.path.exists(file_path):
            logger.debug(f"DOWNLOAD: file_path does not exist: {file_path}")
            return "Файл не найден", 404
        logger.debug(f"DOWNLOAD: sending file {file_path}")
        return send_file(file_path, as_attachment=True, download_name=filename)
    except Exception as e:
        logger.error(f"Ошибка скачивания файла: {e}")
        return "Ошибка скачивания", 500

@app.route('/train')
def train():
    return render_template('train.html')

@app.route('/train_progress')
def train_progress_page():
    return render_template('train_progress.html')

@app.route('/api/train', methods=['POST'])
def api_train():
    try:
        xml_files = request.files.getlist('xml_file')
        zip_files = request.files.getlist('zip_file')
        
        if not xml_files:
            return jsonify({'error': 'Пожалуйста, загрузите XML файлы.'})
        elif not zip_files:
            return jsonify({'error': 'Пожалуйста, загрузите архив(ы) с изображениями.'})
        
        batch_size = int(request.form.get('batch_size', 16))
        epochs = int(request.form.get('epochs', 5))
        
        # Создаем временную папку
        session_dir = tempfile.mkdtemp(prefix='train_session_')
        
        xml_paths = []
        zip_paths = []
        
        # Сохраняем XML файлы
        for xml_file in xml_files:
            if not xml_file.filename:
                continue
            xml_path = os.path.join(session_dir, safe_filename_with_cyrillic(xml_file.filename))
            xml_file.save(xml_path)
            xml_paths.append(xml_path)
        
        # Сохраняем и распаковываем ZIP файлы
        for zip_file in zip_files:
            if not zip_file.filename:
                continue
            zip_path = os.path.join(session_dir, safe_filename_with_cyrillic(zip_file.filename))
            zip_file.save(zip_path)
            zip_paths.append(zip_path)
        
        # Распаковываем ZIP файлы
        for zip_path in zip_paths:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(session_dir)
        
        # Собираем датасет
        xml_path_name_pairs = [(path, os.path.basename(path)) for path in xml_paths]
        settings = load_settings()
        export_format = settings.get('export_format', 'standard')
        manual_review_enabled = False # По умолчанию для обучения
        df = collect_dataset_with_session_dir(xml_path_name_pairs, session_dir, export_format=export_format, manual_review_enabled=manual_review_enabled)
        
        if df.empty:
            return jsonify({'error': 'Не удалось обработать XML файлы!'})
        
        # Загружаем категории пар
        pairs = load_all_category_pairs('category_pairs.yaml')
        visual_pairs = set()
        
        if 'visual' in pairs and isinstance(pairs['visual'], list):
            for pair in pairs['visual']:
                if isinstance(pair, list) and len(pair) == 2:
                    a, b = pair
                    visual_pairs.add((a, b) if a <= b else (b, a))
        
        # Фильтруем только visual пары
        df_visual = df[df.apply(lambda row: get_pair(row) in visual_pairs, axis=1)].copy()
        
        if len(df_visual) == 0:
            return jsonify({'error': 'Нет подходящих коллизий для обучения модели (Approved и Active)!'})
        
        # Проверяем наличие image_file
        if 'image_file' not in df_visual.columns:
            return jsonify({'error': 'Колонка image_file не найдена в данных!'})
        
        # Фильтруем строки с пустыми image_file
        df_visual = df_visual[pd.Series(df_visual['image_file']).notna() & (pd.Series(df_visual['image_file']) != '')]
        
        if len(df_visual) == 0:
            return jsonify({'error': 'Не найдено изображений для обучения!'})
        
        # Создаем модель
        model_name = f'model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pt'
        model_save_path = os.path.join('model', model_name)
        
        # Функция обратного вызова для прогресса
        def progress_callback(epoch, batch, total_batches, train_loss, val_loss, val_acc, f1, recall, precision=None):
            update_train_progress({
                'epoch': epoch,
                'batch': batch,
                'total_batches': total_batches,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'f1': f1,
                'recall': recall,
                'precision': precision
            })
        
        # Запускаем обучение
        metrics = train_model(df_visual, epochs=epochs, batch_size=batch_size, progress_callback=progress_callback)
        
        if metrics is None:
            return jsonify({'error': 'Обучение не запущено: в обучающей выборке только один класс!'})
        
        # Сохраняем статистику
        from collections import Counter
        pair_counts = Counter(pd.DataFrame(df_visual).apply(get_pair, axis=1))
        pairs_with_counts = sorted([[str(p[0]), str(p[1]), int(c)] for p, c in pair_counts.items()], key=lambda x: (-int(x[2] or 0), str(x[0]), str(x[1])))
        stats = {
            'model_file': model_name,
            'train_time': datetime.now().isoformat(),
            'metrics': metrics,
            'category_pairs': pairs_with_counts,
            'epochs': epochs,
            'batch_size': batch_size
        }
        
        stats_path = os.path.join('model', f'{model_name}_stats.json')
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2, default=str)
        
        return jsonify(to_py({
            'success': True,
            'model_file': model_name,
            'metrics': metrics
        }))
        
    except Exception as e:
        logger.error(f"Ошибка обучения: {e}")
        return jsonify({'error': f'Ошибка обучения: {str(e)}'})

@app.route('/api/train_progress')
def api_train_progress():
    return jsonify(train_progress)

@app.route('/cleanup_session', methods=['POST'])
def cleanup_session():
    try:
        session_id = request.form.get('session_id')
        if session_id and session_id in session_dirs:
            session_dir = session_dirs[session_id]
            if os.path.exists(session_dir):
                shutil.rmtree(session_dir)
            del session_dirs[session_id]
            return jsonify({'success': True})
        return jsonify({'error': 'Сессия не найдена'})
    except Exception as e:
        logger.error(f"Ошибка очистки сессии: {e}")
        return jsonify({'error': str(e)})

def to_py(obj):
    import numpy as np
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if hasattr(obj, 'item'):
        return obj.item()
    if isinstance(obj, (list, tuple)):
        return [to_py(x) for x in obj]
    if isinstance(obj, dict):
        return {k: to_py(v) for k, v in obj.items()}
    return str(obj)

if __name__ == '__main__':
    app.run(debug=True) 
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
    """Применяет ручную разметку к DataFrame из файла manual_review.json"""
    review_path = os.path.join(session_dir, 'manual_review.json')
    if not os.path.exists(review_path):
        return df
    
    try:
        import json
        with open(review_path, 'r', encoding='utf-8') as f:
            reviews = json.load(f)
        
        review_map = {r['clash_id']: r['status'] for r in reviews}
        df_updated = df.copy()
        
        for idx, row in df_updated.iterrows():
            cid = row.get('clash_id')
            if cid in review_map:
                status = review_map[cid]
                if status == 'Approved':
                    df_updated.at[idx, 'cv_prediction'] = 0
                    df_updated.at[idx, 'prediction_source'] = 'manual_review'
                elif status == 'Active':
                    df_updated.at[idx, 'cv_prediction'] = 1
                    df_updated.at[idx, 'prediction_source'] = 'manual_review'
                else:  # Reviewed
                    df_updated.at[idx, 'cv_prediction'] = -1
                    df_updated.at[idx, 'prediction_source'] = 'manual_review'
        
        logger.info(f"Применена ручная разметка к {len(reviews)} коллизиям")
        return df_updated
    except Exception as e:
        logger.error(f"Ошибка применения ручной разметки: {e}")
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
        return pd.concat(all_data, ignore_index=True)
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
        
        # Экспортируем результаты
        download_links = []
        stats_per_file = []
        
        # Применяем ручную разметку к общему DataFrame для корректного подсчета статистики
        df_with_manual = apply_manual_review(df, session_dir)
        
        for i, (xml_path, orig_filename) in enumerate(xml_path_name_pairs):
            orig_base_name = os.path.splitext(orig_filename)[0]
            
            if export_format == 'bimstep':
                output_xml = os.path.join(session_dir, f"bimstep_results_{orig_base_name}.xml")
            else:
                output_xml = os.path.join(session_dir, f"cv_results_{orig_base_name}.xml")
            
            # Фильтруем данные по файлу
            df_file = df_with_manual[df_with_manual['source_file'] == orig_filename].copy()
            
            if not df_file.empty:
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
                                row.get('clash_id', ''), 
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
                if export_format == 'bimstep':
                    download_filename = orig_filename  # вместо f"bimstep_results_{orig_base_name}.xml"
                else:
                    download_filename = orig_filename
                
                download_url = url_for('download_file', session_id=session_id, filename=safe_filename_with_cyrillic(download_filename))
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
        
        # --- Детальная статистика по типам разметки ---
        detailed_stats = []
        for i, (xml_path, orig_filename) in enumerate(xml_path_name_pairs):
            orig_base_name = os.path.splitext(orig_filename)[0]
            
            # Фильтруем данные по файлу из DataFrame с примененной ручной разметкой
            df_file = df_with_manual[df_with_manual['source_file'] == orig_filename].copy()
            
            if not df_file.empty:
                # Статистика по алгоритму
                algorithm_approved = len(df_file[(df_file['prediction_source'] == 'algorithm') & (df_file['cv_prediction'] == 0)])
                algorithm_active = len(df_file[(df_file['prediction_source'] == 'algorithm') & (df_file['cv_prediction'] == 1)])
                algorithm_reviewed = len(df_file[(df_file['prediction_source'] == 'algorithm') & (df_file['cv_prediction'] == -1)])
                
                # Статистика по модели
                model_approved = len(df_file[(df_file['prediction_source'] == 'model') & (df_file['cv_prediction'] == 0)])
                model_active = len(df_file[(df_file['prediction_source'] == 'model') & (df_file['cv_prediction'] == 1)])
                model_reviewed = len(df_file[(df_file['prediction_source'] == 'model') & (df_file['cv_prediction'] == -1)])
                
                # Статистика по неопределенным (model_uncertain)
                uncertain_reviewed = len(df_file[(df_file['prediction_source'] == 'model_uncertain') & (df_file['cv_prediction'] == -1)])
                
                # Статистика по ручной разметке
                manual_approved = len(df_file[(df_file['prediction_source'] == 'manual_review') & (df_file['cv_prediction'] == 0)])
                manual_active = len(df_file[(df_file['prediction_source'] == 'manual_review') & (df_file['cv_prediction'] == 1)])
                manual_reviewed = len(df_file[(df_file['prediction_source'] == 'manual_review') & (df_file['cv_prediction'] == -1)])
                
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
                        'reviewed': model_reviewed + uncertain_reviewed  # model_uncertain тоже считается как модель
                    },
                    'manual': {
                        'approved': manual_approved,
                        'active': manual_active,
                        'reviewed': manual_reviewed
                    }
                })
        
        # --- Новый блок: формируем список для ручной разметки ---
        manual_review_collisions = []
        if manual_review_enabled:
            # Используем исходный DataFrame для формирования списка ручной разметки
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
                    logger.debug(f"MANUAL_REVIEW: clash_id={row.get('clash_id', '')}, image_file={rel_image_path}")
                    manual_review_collisions.append({
                        'clash_id': row.get('clash_id', ''),
                        'image_file': rel_image_path,
                        'element1_category': row.get('element1_category', ''),
                        'element2_category': row.get('element2_category', ''),
                        'description': row.get('clash_name', ''),
                        'source_file': row.get('source_file', ''),
                        'session_id': session_id
                    })
        return jsonify(to_py({
            'success': True,
            'session_id': session_id,
            'download_links': download_links,
            'stats_per_file': stats_per_file,
            'stats_total': stats_total,
            'used_images': export_format == 'standard',
            'analysis_settings': analysis_settings,
            'manual_review_collisions': manual_review_collisions,
            'detailed_stats': detailed_stats
        }))
        
    except Exception as e:
        logger.error(f"Ошибка анализа файлов: {e}")
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
        if not session_id or not reviews:
            return jsonify({'error': 'Нет session_id или разметок'}), 400
        session_dir = session_dirs.get(session_id)
        if not session_dir or not os.path.exists(session_dir):
            return jsonify({'error': 'Сессия не найдена'}), 404
        # Сохраняем разметку в файл
        review_path = os.path.join(session_dir, 'manual_review.json')
        import json
        with open(review_path, 'w', encoding='utf-8') as f:
            json.dump(reviews, f, ensure_ascii=False, indent=2)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/updated_stats/<session_id>', methods=['GET'])
def api_updated_stats(session_id):
    """Возвращает обновленную статистику после ручной разметки"""
    try:
        session_dir = session_dirs.get(session_id)
        if not session_dir or not os.path.exists(session_dir):
            return jsonify({'error': 'Сессия не найдена'}), 404
        
        # Находим исходный XML файл
        orig_xml = None
        for f in os.listdir(session_dir):
            if f.endswith('.xml') and not f.startswith(('cv_results_', 'bimstep_results_', 'JournalBimStep')):
                orig_xml = os.path.join(session_dir, f)
                break
        
        if not orig_xml:
            return jsonify({'error': 'Исходный XML файл не найден'}), 404
        
        # Загружаем настройки
        settings = load_settings()
        export_format = settings.get('export_format', 'standard')
        
        # Парсим исходные данные
        df = parse_xml_data(orig_xml, export_format)
        df['source_file'] = os.path.basename(orig_xml)
        
        # Применяем ручную разметку
        df_with_manual = apply_manual_review(df, session_dir)
        
        # Подсчитываем обновленную статистику
        total_approved = (df_with_manual['cv_prediction'] == 0).sum()
        total_active = (df_with_manual['cv_prediction'] == 1).sum()
        total_reviewed = (df_with_manual['cv_prediction'] == -1).sum()
        
        updated_stats = {
            'total_collisions': len(df_with_manual),
            'total_approved': total_approved,
            'total_active': total_active,
            'total_reviewed': total_reviewed
        }
        
        return jsonify(updated_stats)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

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
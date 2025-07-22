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
import threading
import urllib.parse
import glob
import pickle
from concurrent.futures import ThreadPoolExecutor
import time

from core.xml_utils import (
    load_all_category_pairs, get_pair, parse_xml_data, export_to_bimstep_xml, export_to_xml, add_bimstep_journal_entry
)
from core.image_utils import find_image_by_name, get_relative_image_path, get_absolute_image_path_optimized, build_image_index_with_cache
from ml.model import create_model
from ml.dataset import CollisionImageDataset, create_transforms
from ml.inference import predict, fill_xml_fields
from ml.train import train_model
from web.utils import safe_filename_with_cyrillic, load_settings, save_settings
from web.progress import update_train_progress, load_train_progress

logger = logging.getLogger(__name__)

# Глобальная переменная для хранения пути к последней временной папке обучения
last_train_temp_dir = None

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
#Словарь для хранения прогресса анализа
analysis_progress = {}

def update_analysis_progress(session_id, key, status):
    """Обновляет статус конкретного этапа анализа."""
    if session_id in analysis_progress:
        # Устанавливаем статус для указанного ключа
        for stage in analysis_progress[session_id].get('stage_statuses', []):
            if stage['key'] == key:
                stage['status'] = status
                break
        
        # Если этап перешел в 'in_progress', все предыдущие должны быть 'done'
        if status == 'in_progress':
            stage_order = ['algorithm', 'model', 'manual']
            current_index = stage_order.index(key)
            for i in range(current_index):
                prev_key = stage_order[i]
                for stage in analysis_progress[session_id].get('stage_statuses', []):
                    if stage['key'] == prev_key and stage['enabled']:
                        stage['status'] = 'done'


def run_analysis_background(session_id, session_dir, xml_path_name_pairs, zip_paths, settings):
    """Эта функция выполняется в фоновом потоке для обработки анализа."""
    with app.app_context():
        try:
            import pandas as pd
            import torch
            from core.image_utils import build_image_index_with_cache, resolve_images_vectorized_series
            from core.xml_utils import get_pairs_vectorized, filter_valid_images
            from ml.model import get_cached_model, create_model, get_model_info
            from ml.dataset import create_transforms
            
            export_format = settings.get('export_format', 'standard')
            inference_mode = settings.get('inference_mode', 'model')
            low_confidence = settings.get('low_confidence', 0.3)
            high_confidence = settings.get('high_confidence', 0.7)
            manual_review_enabled = settings.get('manual_review_enabled', False)
            
            # --- Распаковка файлов ---
            if zip_paths:
                for zip_path in zip_paths:
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(session_dir)

            # --- Индексация изображений ---
            image_index = get_or_build_image_index(session_dir)

            # --- Парсинг XML ---
            def parse_one(args):
                xml_path, safe_filename = args
                df_part = parse_xml_data(xml_path, export_format=export_format)
                if df_part.empty: return None
                df_part['source_file'] = safe_filename
                return df_part
            
            with ThreadPoolExecutor() as executor:
                dfs = list(executor.map(parse_one, xml_path_name_pairs))
            dfs = [df for df in dfs if df is not None]

            if not dfs:
                raise ValueError("Не удалось обработать XML файлы!")
            df = pd.concat(dfs, ignore_index=True)

            # --- Алгоритмическая разметка ---
            update_analysis_progress(session_id, 'algorithm', 'in_progress')
            # (логика алгоритмической разметки...)
            pairs = load_all_category_pairs('category_pairs.yaml')
            can_pairs = set(tuple(sorted(p)) for p in pairs.get('can', []))
            cannot_pairs = set(tuple(sorted(p)) for p in pairs.get('cannot', []))
            visual_pairs = set(tuple(sorted(p)) for p in pairs.get('visual', []))
            pairs_series = get_pairs_vectorized(df)
            mask_can = pairs_series.isin(list(can_pairs))
            mask_cannot = pairs_series.isin(list(cannot_pairs))
            mask_visual = pairs_series.isin(list(visual_pairs))
            mask_unknown = ~(mask_can | mask_cannot | mask_visual)
            df['cv_prediction'] = -1
            df['cv_confidence'] = 0.5
            df['prediction_source'] = 'algorithm'
            df['cv_status'] = 'Reviewed'
            df.loc[mask_can, ['cv_prediction', 'cv_confidence', 'prediction_source', 'cv_status']] = [0, 1.0, 'algorithm', 'Approved']
            df.loc[mask_cannot, ['cv_prediction', 'cv_confidence', 'prediction_source', 'cv_status']] = [1, 1.0, 'algorithm', 'Active']
            update_analysis_progress(session_id, 'algorithm', 'done')
            
            # --- Поиск изображений ---
            need_images_mask = (df['cv_prediction'] == -1)
            if need_images_mask.any():
                df_need_images = df.loc[need_images_mask]
                if 'image_href' in df_need_images.columns:
                    df.loc[need_images_mask, 'image_file'] = resolve_images_vectorized_series(df_need_images['image_href'], image_index, session_dir, parallel=True)
                else:
                     df.loc[need_images_mask, 'image_file'] = ''
                df.loc[need_images_mask, :] = filter_valid_images(df.loc[need_images_mask, :])

            # --- Инференс модели ---
            if inference_mode == 'model':
                visual_mask_combined = mask_visual | mask_unknown
                if visual_mask_combined.any():
                    update_analysis_progress(session_id, 'model', 'in_progress')
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    visual_df = df.loc[visual_mask_combined].copy()
                    transform = create_transforms(is_training=False)
                    model_path = os.path.join('model', settings.get('model_file', 'model_clashmark.pt'))
                    
                    def get_model_type_for_file(model_file):
                        log_path = os.path.join('model', 'model_train_log.json')
                        if model_file and os.path.exists(log_path):
                            with open(log_path, 'r', encoding='utf-8') as f: logs = json.load(f)
                            if isinstance(logs, dict): logs = [logs]
                            for entry in logs:
                                if entry.get('model_file') == model_file:
                                    mt = entry.get('model_type', 'mobilenet_v3_small')
                                    mapping = {'MobileNetV3 Small': 'mobilenet_v3_small', 'EfficientNet-B0': 'efficientnet_b0', 'ResNet 18': 'resnet18', 'MobileNetV2': 'mobilenet_v2'}
                                    return mapping.get(mt, mt)
                        return settings.get('model_type', 'mobilenet_v3_small')

                    model_type = get_model_type_for_file(settings.get('model_file')) or 'mobilenet_v3_small'
                    model = get_cached_model(device, model_path, model_type, True)
                    model.eval()
                    visual_pred_df = predict(
                        model=model,
                        device=device,
                        df=visual_df,
                        transform=transform,
                        batch_size=16,
                        low_confidence_threshold=low_confidence,
                        high_confidence_threshold=high_confidence,
                        session_dir=session_dir
                    )

                    visual_indices = visual_df.index
                    df.loc[visual_indices, 'cv_prediction'] = visual_pred_df['cv_prediction'].values
                    df.loc[visual_indices, 'cv_confidence'] = visual_pred_df['cv_confidence'].values
                    df.loc[visual_indices, 'prediction_source'] = visual_pred_df['prediction_source'].values
                    df.loc[visual_indices, 'cv_status'] = visual_pred_df['cv_status'].values
                    
                    update_analysis_progress(session_id, 'model', 'done')
            
            # ... (Остальная логика для подготовки ответа, как была в analyze_files)
            # Сохраняем промежуточный CSV
            df.to_csv(os.path.join(session_dir, 'df_with_inference.csv'), index=False, encoding='utf-8')
            
            # --- Подготовка финального ответа ---
            # Эта логика теперь здесь, в конце фоновой задачи
            df_with_manual = apply_manual_review(df.copy(), session_dir)
            download_links = []
            stats_per_file = []
            detailed_stats = []

            for xml_path, orig_filename in xml_path_name_pairs:
                df_file = df_with_manual[df_with_manual['source_file'] == orig_filename].copy()
                if df_file.empty:
                    continue

                orig_base_name = os.path.splitext(orig_filename)[0]
                
                if export_format == 'bimstep':
                    output_xml = os.path.join(session_dir, f"bimstep_results_{orig_base_name}.xml")
                    export_to_bimstep_xml(df_file, output_xml, xml_path)
                    # Добавляем запись в журнал для каждой строки
                    for _, row in df_file.iterrows():
                         if row.get('cv_prediction') is not None:
                            status_map = {0: 'approved', 1: 'active', -1: 'reviewed'}
                            status = status_map.get(row.get('cv_prediction'), 'reviewed')
                            add_bimstep_journal_entry(
                                row.get('clash_uid', ''), 'Разметка ClashMark', 'Разметка ClashMark',
                                status=status, session_dir=session_dir
                            )
                else: # 'standard'
                    output_xml = os.path.join(session_dir, f"cv_results_{orig_base_name}.xml")
                    export_to_xml(df_file, output_xml, xml_path)

                download_filename = f"{orig_base_name}.xml" # Use original name for download
                download_url = f"/download/{session_id}/{urllib.parse.quote(os.path.basename(output_xml))}"
                download_links.append({'name': download_filename, 'url': download_url})

            
            # После цикла добавляем ссылку на журнал, если он существует
            if export_format == 'bimstep':
                journal_path = os.path.join(session_dir, 'JournalBimStep.xml')
                if os.path.exists(journal_path):
                    journal_url = f"/download/{session_id}/JournalBimStep.xml"
                    if not any(d['name'] == 'JournalBimStep.xml' for d in download_links):
                        download_links.append({'name': 'JournalBimStep.xml', 'url': journal_url})

            stats_total = {'total_files': len(xml_path_name_pairs), 'total_collisions': len(df_with_manual),
                           'total_approved': int((df_with_manual['cv_prediction'] == 0).sum()),
                           'total_active': int((df_with_manual['cv_prediction'] == 1).sum()),
                           'total_reviewed': int((df_with_manual['cv_prediction'] == -1).sum())}

            manual_review_collisions = []
            if manual_review_enabled:
                # ... (логика формирования списка для ручной разметки) ...
                for _, row in df.iterrows():
                    if row.get('cv_prediction') == -1 and pd.notna(row.get('image_file')):
                        manual_review_collisions.append({
                            'clash_uid': row.get('clash_uid', ''),
                            'image_file': row.get('image_file', ''), # Просто передаем абсолютный путь
                            'element1_category': row.get('element1_category', ''),
                            'element2_category': row.get('element2_category', ''),
                            'description': row.get('clash_name', ''),
                            'source_file': row.get('source_file', ''),
                            'session_id': session_id
                        })
            
            final_result = {
                'success': True,
                'download_links': download_links,
                'stats_total': stats_total,
                'manual_review_collisions': manual_review_collisions,
                'detailed_stats': detailed_stats
            }
            analysis_progress[session_id]['status'] = 'finished'
            analysis_progress[session_id]['result'] = final_result
            
        except Exception as e:
            logger.error(f"Ошибка в фоновом анализе для сессии {session_id}: {e}", exc_info=True)
            analysis_progress[session_id]['status'] = 'error'
            analysis_progress[session_id]['error'] = f'Внутренняя ошибка сервера: {e}'


def apply_manual_review(df, session_dir):
    """Применяет ручную разметку к DataFrame из файла manual_review.json, обновляя только строки с clash_uid из разметки, остальные не трогает."""
    review_path = os.path.join(session_dir, 'manual_review.json')
    if not os.path.exists(review_path):
        logger.info(f"Файл ручной разметки не найден: {review_path}")
        return df
    try:
        import json
        # Пробуем разные способы чтения файла
        reviews = None
        try:
            with open(review_path, 'r', encoding='utf-8') as f:
                reviews = json.load(f)
        except UnicodeDecodeError:
            # Пробуем с другой кодировкой
            with open(review_path, 'r', encoding='latin-1') as f:
                reviews = json.load(f)
        except Exception as e:
            logger.error(f"[apply_manual_review] Ошибка чтения файла: {e}")
            return df
        
        if not reviews:
            logger.warning(f"[apply_manual_review] Файл пустой или не содержит данных")
            return df
            
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
        logger.error(f"[apply_manual_review] Ошибка применения ручной разметки: {e}")
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
                    # Используем оптимизированную функцию поиска
                    image_path = get_absolute_image_path_optimized(image_href, session_dir)
                    if not image_path:
                        # Fallback к старому методу
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
            # Добавляем новые настройки
            settings['model_type'] = request.form.get('model_type', 'mobilenet_v3_small')
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
                    # Удаляем записи из model_train_log.json
                    log_path = os.path.join('model', 'model_train_log.json')
                    if os.path.exists(log_path):
                        with open(log_path, 'r', encoding='utf-8') as f:
                            logs = json.load(f)
                        if isinstance(logs, dict):
                            logs = [logs]
                        logs = [entry for entry in logs if entry.get('model_file') != model_file]
                        with open(log_path, 'w', encoding='utf-8') as f:
                            json.dump(logs, f, ensure_ascii=False, indent=2)
                    flash(f'Модель {model_file} удалена!', 'success')
                except Exception as e:
                    flash(f'Ошибка удаления модели: {e}', 'error')
            else:
                flash('Нельзя удалить основную модель!', 'error')
                
        elif action == 'clear_cache':
            import tempfile
            import glob
            removed = 0
            errors = []
            # Удаляем только из системного tmp
            tmp_dir = tempfile.gettempdir()
            for pattern in ['analysis_session_*', 'train_session_*']:
                for folder in glob.glob(os.path.join(tmp_dir, pattern)):
                    try:
                        if os.path.isdir(folder):
                            shutil.rmtree(folder)
                            removed += 1
                    except Exception as e:
                        errors.append(f"{folder}: {e}")
            if errors:
                flash(f'Удалено {removed} папок, ошибки: {errors}', 'error')
            else:
                flash(f'Удалено {removed} временных папок!', 'success')
    
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
    # Загрузим карту model_file -> model_type из model_train_log.json
    model_types = {}
    log_path = os.path.join('model', 'model_train_log.json')
    if os.path.exists(log_path):
        try:
            with open(log_path, 'r', encoding='utf-8') as f:
                logs = json.load(f)
            if isinstance(logs, dict):
                logs = [logs]
            for entry in logs:
                if 'model_file' in entry and 'model_type' in entry:
                    model_types[entry['model_file']] = entry['model_type']
        except Exception as e:
            logger.error(f"Ошибка чтения model_train_log.json: {e}")
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
                    # Добавляем архитектуру из train_log, если есть
                    stats['model_type'] = model_types.get(model_file, stats.get('model_type', '—'))
                    model_metrics[model_file] = stats
            except Exception as e:
                logger.error(f"Ошибка загрузки статистики для {model_file}: {e}")
    
    return render_template('settings.html', settings=settings, model_files=model_files, model_metrics=model_metrics)

@app.route('/api/settings', methods=['GET'])
def api_settings():
    settings = load_settings()
    if 'model_type' in settings:
        settings['model_type_pretty'] = prettify_model_type(settings['model_type'])
    return jsonify(to_py(settings))

def get_or_build_image_index(session_dir):
    """
    Кэширует индекс изображений в session_dir/image_index.pkl
    """
    cache_path = os.path.join(session_dir, 'image_index.pkl')
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except Exception:
            pass
    index = build_image_index_with_cache(session_dir)
    try:
        with open(cache_path, 'wb') as f:
            pickle.dump(index, f)
    except Exception:
        pass
    return index

@app.route('/api/analysis_progress/<session_id>')
def api_analysis_progress(session_id):
    """Возвращает текущий прогресс анализа."""
    progress = analysis_progress.get(session_id, {})
    return jsonify(to_py(progress))

@app.route('/analyze', methods=['POST'])
def analyze_files():
    """Запускает анализ в фоновом режиме и немедленно возвращает session_id."""
    try:
        xml_files = request.files.getlist('xml_file')
        zip_files = request.files.getlist('images_zip')
        
        settings = load_settings()
        inference_mode = settings.get('inference_mode', 'model')
        manual_review_enabled = settings.get('manual_review_enabled', False)

        if not xml_files:
            return jsonify({'error': 'Не выбраны XML файлы!'})

        session_dir = tempfile.mkdtemp(prefix='analysis_session_')
        session_id = os.path.basename(session_dir)
        session_dirs[session_id] = session_dir

        xml_path_name_pairs = []
        for xml_file in xml_files:
            if not xml_file.filename: continue
            safe_filename = safe_filename_with_cyrillic(xml_file.filename)
            xml_path = os.path.join(session_dir, safe_filename)
            xml_file.save(xml_path)
            xml_path_name_pairs.append((xml_path, safe_filename))
        
        zip_paths = []
        if zip_files:
            for zip_file in zip_files:
                if not zip_file.filename: continue
                zip_path = os.path.join(session_dir, safe_filename_with_cyrillic(zip_file.filename))
                zip_file.save(zip_path)
                zip_paths.append(zip_path)

        # Инициализация прогресса
        stages = [
            {'key': 'algorithm', 'label': 'Разметка алгоритмом', 'enabled': True, 'status': 'pending'},
            {'key': 'model', 'label': 'Разметка моделью', 'enabled': inference_mode == 'model', 'status': 'pending'},
            {'key': 'manual', 'label': 'Ручная разметка', 'enabled': manual_review_enabled, 'status': 'pending'}
        ]
        analysis_progress[session_id] = {
            'status': 'running',
            'stage_statuses': [s for s in stages if s['enabled']]
        }

        # Запуск фоновой задачи
        thread = threading.Thread(target=run_analysis_background, args=(
            session_id, session_dir, xml_path_name_pairs, zip_paths, settings
        ))
        thread.start()

        return jsonify({'success': True, 'session_id': session_id})
        
    except Exception as e:
        logger.error(f"Ошибка при запуске анализа: {e}", exc_info=True)
        return jsonify({'error': f'Ошибка запуска анализа: {str(e)}'})

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
            # Сначала пробуем стандартный способ
            with open(review_path, 'w', encoding='utf-8') as f:
                json.dump(reviews, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"[manual_review] Первая попытка записи не удалась: {e}")
            try:
                # Альтернативный способ для Windows
                import tempfile
                temp_path = os.path.join(session_dir, 'manual_review_temp.json')
                with open(temp_path, 'w', encoding='utf-8') as f:
                    json.dump(reviews, f, ensure_ascii=False, indent=2)
                # Переименовываем файл
                if os.path.exists(review_path):
                    os.remove(review_path)
                os.rename(temp_path, review_path)
            except Exception as e2:
                logger.error(f"[manual_review] Вторая попытка записи не удалась: {e2}")
                try:
                    # Последняя попытка - запись без форматирования
                    with open(review_path, 'w', encoding='utf-8') as f:
                        json.dump(reviews, f, ensure_ascii=False)
                except Exception as e3:
                    logger.error(f"[manual_review] Все попытки записи не удались: {e3}")
                    return jsonify({'error': f'Ошибка записи manual_review.json: {e3}'}), 500
        
        # Проверяем, что файл действительно создался
        if not os.path.exists(review_path):
            logger.error(f"[manual_review] Файл не создался после записи: {review_path}")
            return jsonify({'error': 'Файл ручной разметки не был создан'}), 500
        
        # Проверяем содержимое файла
        try:
            with open(review_path, 'r', encoding='utf-8') as f:
                saved_reviews = json.load(f)
            if len(saved_reviews) != len(reviews):
                logger.warning(f"[manual_review] Количество сохраненных разметок ({len(saved_reviews)}) не совпадает с отправленными ({len(reviews)})")
        except Exception as e:
            logger.error(f"[manual_review] Ошибка проверки сохраненного файла: {e}")
        
        logger.info(f"Ручная разметка сохранена в {review_path}")
        logger.info(f"Статистика сохраненной разметки: {status_counts}")
        # --- Новый блок: применяем ручную разметку, пересчитываем статистику и переэкспортируем XML ---
        import pandas as pd
        df_path = os.path.join(session_dir, 'df_with_inference.csv')
        df_combined = None
        if os.path.exists(df_path):
            # Пробуем разные способы чтения CSV
            try:
                df_combined = pd.read_csv(df_path, encoding='utf-8')
            except UnicodeDecodeError:
                try:
                    df_combined = pd.read_csv(df_path, encoding='latin-1')
                except Exception as e:
                    logger.error(f"[manual_review] Ошибка чтения CSV: {e}")
                    return jsonify({'error': 'Ошибка чтения данных анализа'}), 500
            
            if df_combined is None or df_combined.empty:
                logger.error(f"[manual_review] DataFrame пустой после чтения")
                return jsonify({'error': 'Данные анализа повреждены'}), 500
        else:
            logger.error(f"[manual_review] Файл с инференсом не найден: {df_path}")
            return jsonify({'error': 'Нет данных для обновления статистики. Проведите анализ заново.'}), 400
        
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
                if export_format == 'bimstep' or export_format == 'standard':
                    output_xml = os.path.join(session_dir, f"{orig_base_name}.xml")
                    export_to_bimstep_xml(df_file, output_xml, orig_xml)
                else:
                    output_xml = os.path.join(session_dir, f"cv_results_{orig_base_name}.xml")
                    export_to_xml(df_file, output_xml, orig_xml)
                
                # Создаем ссылки для скачивания
                # Для всех форматов используем исходное имя файла
                download_filename = f"{orig_base_name}.xml"
                
                download_url = f"/download/{session_id}/{urllib.parse.quote(os.path.basename(output_xml))}"
                download_links.append({'name': download_filename, 'url': download_url})
                if export_format == 'bimstep' or export_format == 'standard':
                    journal_path = os.path.join(session_dir, 'JournalBimStep.xml')
                    if os.path.exists(journal_path):
                        journal_download_url = f"/download/{session_id}/JournalBimStep.xml"
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
        try:
            df_with_manual.to_csv(df_path, index=False, encoding='utf-8')
        except Exception as e:
            logger.warning(f"[manual_review] Ошибка сохранения CSV: {e}")
            try:
                # Альтернативный способ сохранения
                temp_csv_path = os.path.join(session_dir, 'df_with_inference_temp.csv')
                df_with_manual.to_csv(temp_csv_path, index=False, encoding='utf-8')
                if os.path.exists(df_path):
                    os.remove(df_path)
                os.rename(temp_csv_path, df_path)
            except Exception as e2:
                logger.error(f"[manual_review] Не удалось сохранить CSV: {e2}")
        
        return jsonify({'success': True, 'status_counts': status_counts, 'download_links': download_links, 'detailed_stats': detailed_stats})
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
        
        # Пробуем разные способы чтения CSV
        df_combined = None
        try:
            df_combined = pd.read_csv(df_path, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                df_combined = pd.read_csv(df_path, encoding='latin-1')
            except Exception as e:
                logger.error(f"[api_updated_stats] Ошибка чтения CSV: {e}")
                return jsonify({'error': 'Ошибка чтения данных анализа'}), 500
        
        if df_combined is None or df_combined.empty:
            logger.error(f"[api_updated_stats] DataFrame пустой после чтения")
            return jsonify({'error': 'Данные анализа повреждены'}), 500
        
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
                if export_format == 'bimstep' or export_format == 'standard':
                    output_xml = os.path.join(session_dir, f"{orig_base_name}.xml")
                    export_to_bimstep_xml(df_file, output_xml, orig_xml)
                else:
                    output_xml = os.path.join(session_dir, f"cv_results_{orig_base_name}.xml")
                    export_to_xml(df_file, output_xml, orig_xml)
                logger.info(f"Переэкспортирован файл {orig_filename} с учетом ручной разметки")
                # --- формируем ссылку для скачивания ---
                download_url = f"/download/{session_id}/{urllib.parse.quote(os.path.basename(output_xml))}"
                download_links.append({'name': os.path.basename(output_xml), 'url': download_url})
                # Для BIM Step добавляем JournalBimStep.xml
                if export_format == 'bimstep' or export_format == 'standard':
                    journal_path = os.path.join(session_dir, 'JournalBimStep.xml')
                    if os.path.exists(journal_path):
                        journal_download_url = f"/download/{session_id}/JournalBimStep.xml"
                        download_links.append({'name': 'JournalBimStep.xml', 'url': journal_download_url})
        # --- Конец блока ---
        # Перезаписываем df_with_inference.csv актуальным DataFrame с ручной разметкой
        try:
            df_with_manual.to_csv(os.path.join(session_dir, 'df_with_inference.csv'), index=False, encoding='utf-8')
        except Exception as e:
            logger.warning(f"[api_updated_stats] Ошибка сохранения CSV: {e}")
            try:
                # Альтернативный способ сохранения
                temp_csv_path = os.path.join(session_dir, 'df_with_inference_temp.csv')
                df_with_manual.to_csv(temp_csv_path, index=False, encoding='utf-8')
                csv_path = os.path.join(session_dir, 'df_with_inference.csv')
                if os.path.exists(csv_path):
                    os.remove(csv_path)
                os.rename(temp_csv_path, csv_path)
            except Exception as e2:
                logger.error(f"[api_updated_stats] Не удалось сохранить CSV: {e2}")
        
        logger.info(f"Возвращаем обновленную статистику: {full_response}")
        logger.info(f"Детальная статистика содержит {len(detailed_stats)} файлов")
        # --- добавляем download_links в ответ ---
        full_response['download_links'] = download_links
        return jsonify(to_py(full_response))
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
        
        decoded_filename = urllib.parse.unquote(filename)
        
        # Ищем файл рекурсивно в директории сессии
        file_path = None
        for root, dirs, files in os.walk(session_dir):
            if decoded_filename in files:
                file_path = os.path.join(root, decoded_filename)
                break
        
        if not file_path:
            logger.debug(f"DOWNLOAD: file '{decoded_filename}' not found anywhere in '{session_dir}'")
            return "Файл не найден", 404

        image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.svg'}
        _, ext = os.path.splitext(decoded_filename)
        as_attachment = ext.lower() not in image_extensions

        logger.debug(f"DOWNLOAD: sending file {file_path}, as_attachment={as_attachment}")
        return send_file(file_path, as_attachment=as_attachment, download_name=os.path.basename(file_path))
    except Exception as e:
        logger.error(f"Ошибка скачивания файла: {e}")
        return "Ошибка скачивания", 500

@app.route('/train')
def train():
    from web.utils import load_settings
    settings = load_settings()
    return render_template('train.html', settings=settings)

@app.route('/train_progress')
def train_progress_page():
    from web.progress import load_train_progress
    global last_train_temp_dir
    progress = {}
    if last_train_temp_dir:
        progress = load_train_progress(last_train_temp_dir)
    return render_template('train_progress.html', progress=progress)

@app.route('/api/train', methods=['POST'])
def api_train():
    global last_train_temp_dir
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
        last_train_temp_dir = session_dir  # Сохраняем путь для прогресса
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
        for zip_path in zip_paths:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(session_dir)
        # Определяем папку с изображениями
        bsimages_dir = os.path.join(session_dir, 'BSImages')
        has_bsimages = os.path.exists(bsimages_dir) and os.path.isdir(bsimages_dir)
        settings = load_settings()
        export_format = settings.get('export_format', 'standard')
        manual_review_enabled = False # По умолчанию для обучения
        images_dirs = []
        if export_format == 'bimstep':
            if has_bsimages:
                images_dirs = [bsimages_dir, session_dir]
            else:
                images_dirs = [session_dir]
        else:  # standard
            if has_bsimages:
                images_dirs = [bsimages_dir, session_dir]
            else:
                images_dirs = [session_dir]
        from core.xml_utils import parse_xml_data
        import pandas as pd
        all_dataframes = []
        for xml_path in xml_paths:
            df = parse_xml_data(xml_path, export_format=export_format)
            if len(df) == 0:
                continue
            def find_img(href):
                import os
                from core.image_utils import find_image_by_name
                for d in images_dirs:
                    path = find_image_by_name(href, d) if href else None
                    if path:
                        return path
                return None
            df['image_file'] = df['image_href'].apply(find_img)
            df['source_file'] = os.path.basename(xml_path)
            all_dataframes.append(df)
        if not all_dataframes:
            return jsonify({'error': 'Не удалось обработать XML файлы!'})
        combined_df = pd.concat(all_dataframes, ignore_index=True)
        df_with_images = combined_df[combined_df['image_file'].notna() & combined_df['image_file'].apply(lambda x: x is not None)]
        if not isinstance(df_with_images, pd.DataFrame):
            df_with_images = pd.DataFrame(df_with_images)
        # Проверяем наличие колонки IsResolved и фильтруем данные
        if 'IsResolved' in df_with_images.columns:
            df_with_images = df_with_images[df_with_images['IsResolved'].isin([0, 1])].copy()
        else:
            # Если колонки IsResolved нет, создаем её на основе status
            df_with_images['IsResolved'] = df_with_images['status'].apply(
                lambda x: 1 if x == 'Active' or x == 'Активн.' else (0 if x == 'Approved' or x == 'Подтверждено' else -1)
            )
            df_with_images = df_with_images[df_with_images['IsResolved'].isin([0, 1])].copy()
        
        df = df_with_images
        if df.empty:
            return jsonify({'error': 'Не найдено изображений для обучения!'})
        pairs = load_all_category_pairs('category_pairs.yaml')
        visual_pairs = set()
        if 'visual' in pairs and isinstance(pairs['visual'], list):
            for pair in pairs['visual']:
                if isinstance(pair, list) and len(pair) == 2:
                    a, b = pair
                    visual_pairs.add((a, b) if a <= b else (b, a))
        df_visual = df[df.apply(lambda row: get_pair(row) in visual_pairs, axis=1)].copy()
        # Убедимся, что df_visual — DataFrame
        if not isinstance(df_visual, pd.DataFrame):
            df_visual = pd.DataFrame(df_visual)
        # --- АУГМЕНТАЦИЯ КЛАССА-МЕНЬШИНСТВА ---
        import numpy as np
        from PIL import Image
        from PIL.Image import Transpose
        # Определяем метку: 0 — Approved, 1 — Active
        if not isinstance(df_visual, pd.DataFrame):
            df_visual = pd.DataFrame(df_visual)
        
        # Проверяем наличие колонки status и создаем label
        if 'status' in df_visual.columns:
            df_visual['label'] = df_visual['status'].apply(
                lambda x: 0 if x == 'Approved' or x == 'Подтверждено' else (1 if x == 'Active' or x == 'Активн.' else None)
            )
        elif 'IsResolved' in df_visual.columns:
            # Если нет status, но есть IsResolved, используем его
            df_visual['label'] = df_visual['IsResolved'].apply(
                lambda x: 0 if x == 0 else (1 if x == 1 else None)
            )
        else:
            # Если нет ни status, ни IsResolved, создаем на основе resultstatus
            df_visual['label'] = df_visual['resultstatus'].apply(
                lambda x: 0 if x == 'Подтверждено' else (1 if x == 'Активн.' else None)
            )
        class_counts = df_visual['label'].value_counts().to_dict()
        min_class = min(class_counts, key=lambda k: class_counts[k])
        max_class = max(class_counts, key=lambda k: class_counts[k])
        n_min = class_counts[min_class]
        n_max = class_counts[max_class]
        n_to_add = n_max - n_min
        AUG_DIR = os.path.join(session_dir, 'BSImages_aug')
        os.makedirs(AUG_DIR, exist_ok=True)
        def augment_and_save(row, aug_dir, n_aug):
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
        n_min_rows = df_visual[df_visual['label'] == min_class]
        if not isinstance(n_min_rows, pd.DataFrame):
            n_min_rows = pd.DataFrame(n_min_rows)
        if n_min > 0 and n_to_add > 0:
            n_aug_per_row = int(np.ceil(n_to_add / n_min))
            augmented_rows = []
            for idx, row in n_min_rows.iterrows():
                n_aug = min(n_aug_per_row, n_to_add - len(augmented_rows))
                if n_aug <= 0:
                    break
                augmented_rows.extend(augment_and_save(row, AUG_DIR, n_aug))
            df_visual = pd.concat([df_visual, pd.DataFrame(augmented_rows)], ignore_index=True)
        if 'label' in df_visual.columns:
            df_visual = df_visual.drop(columns=['label'])
        # Добавить папку с аугментированными изображениями в images_dirs
        if AUG_DIR not in images_dirs:
            images_dirs.insert(0, AUG_DIR)
        model_name = f'model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pt'
        model_save_path = os.path.join('model', model_name)
        # Инициализация прогресса обучения до запуска потока
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
        }, session_dir)
        def progress_callback(epoch, batch, total_batches, train_loss, val_loss, val_acc, f1, recall, precision=None, update_metrics=True):
            prog = load_train_progress(session_dir)
            # Инициализация массивов метрик, если их нет
            for key in ['train_losses', 'val_losses', 'val_accuracies', 'val_f1s', 'val_recalls', 'val_precisions']:
                if 'metrics' not in prog or key not in prog['metrics']:
                    prog.setdefault('metrics', {})[key] = []
            # Добавлять значения в массивы только если update_metrics=True (раз в эпоху)
            if update_metrics:
                prog['metrics']['train_losses'].append(float(train_loss))
                prog['metrics']['val_losses'].append(float(val_loss) if val_loss is not None else 0)
                prog['metrics']['val_accuracies'].append(float(val_acc) if val_acc is not None else 0)
                prog['metrics']['val_f1s'].append(float(f1) if f1 is not None else 0)
                prog['metrics']['val_recalls'].append(float(recall) if recall is not None else 0)
                prog['metrics']['val_precisions'].append(float(precision) if precision is not None else 0)
            # last_* значения обновлять всегда
            prog.update({
                'status': 'in_progress',
                'started': True,
                'epoch': epoch + 1,
                'batch': batch + 1,
                'total_epochs': epochs,
                'total_batches': total_batches,
                'loss': float(train_loss),
                'metrics': {
                    **prog.get('metrics', {}),
                    'last_train_loss': float(train_loss),
                    'last_val_loss': float(val_loss) if val_loss is not None else '',
                    'last_val_acc': float(val_acc) if val_acc is not None else '',
                    'last_f1': float(f1) if f1 is not None else '',
                    'last_recall': float(recall) if recall is not None else '',
                    'last_precision': float(precision) if precision is not None else ''
                }
            })
            update_train_progress(prog, session_dir)
        model_type = request.form.get('model_type', 'mobilenet_v3_small')
        pretty_model_type = prettify_model_type(model_type)
        def train_job(model_type, pretty_model_type):
            try:
                # Убеждаемся, что df_visual содержит необходимые колонки
                if 'IsResolved' not in df_visual.columns:
                    # Создаем IsResolved на основе доступных данных
                    if 'status' in df_visual.columns:
                        df_visual['IsResolved'] = df_visual['status'].apply(
                            lambda x: 1 if x == 'Active' or x == 'Активн.' else (0 if x == 'Approved' or x == 'Подтверждено' else -1)
                        )
                    elif 'resultstatus' in df_visual.columns:
                        df_visual['IsResolved'] = df_visual['resultstatus'].apply(
                            lambda x: 1 if x == 'Активн.' else (0 if x == 'Подтверждено' else -1)
                        )
                    else:
                        update_train_progress({'status': 'error', 'log': 'Не найдена колонка IsResolved, status или resultstatus для обучения'}, session_dir)
                        return
                # Фильтруем только нужные классы
                df_visual_filtered = df_visual[df_visual['IsResolved'].isin([0, 1])].copy()
                if len(df_visual_filtered) == 0:
                    update_train_progress({'status': 'error', 'log': 'Не найдено данных для обучения (требуются классы 0 и 1)'}, session_dir)
                    return
                metrics = train_model(df_visual_filtered, epochs=epochs, batch_size=batch_size, progress_callback=progress_callback, model_filename=model_name, model_type=model_type)
                if metrics is None:
                    update_train_progress({'status': 'error', 'log': 'Обучение не запущено: в обучающей выборке только один класс!'}, session_dir)
                    return
                from collections import Counter
                pair_counts = Counter(pd.DataFrame(df_visual).apply(get_pair, axis=1))
                pairs_with_counts = sorted([[str(p[0]), str(p[1]), int(c)] for p, c in pair_counts.items()], key=lambda x: (-int(x[2] or 0), str(x[0]), str(x[1])))
                stats = {
                    'model_file': model_name,
                    'train_time': datetime.now().isoformat(),
                    'metrics': metrics,
                    'category_pairs': pairs_with_counts,
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'model_type': pretty_model_type
                }
                stats_path = os.path.join('model', f'{model_name}_stats.json')
                with open(stats_path, 'w', encoding='utf-8') as f:
                    json.dump(stats, f, ensure_ascii=False, indent=2, default=str)
                # --- Запись в model_train_log.json ---
                log_path = os.path.join('model', 'model_train_log.json')
                log_entry = {
                    "train_time": stats['train_time'],
                    "model_file": stats['model_file'],
                    "final_accuracy": stats['metrics'].get('final_accuracy'),
                    "final_f1": stats['metrics'].get('final_f1'),
                    "final_recall": stats['metrics'].get('final_recall'),
                    "final_precision": stats['metrics'].get('final_precision'),
                    "confusion_matrix": stats['metrics'].get('confusion_matrix'),
                    "epochs": stats['epochs'],
                    "batch_size": stats['batch_size'],
                    "model_type": pretty_model_type
                }
                if os.path.exists(log_path):
                    with open(log_path, 'r', encoding='utf-8') as f:
                        logs = json.load(f)
                    if isinstance(logs, dict):
                        logs = [logs]
                    logs = [entry for entry in logs if entry.get('model_file') != model_name]
                else:
                    logs = []
                logs.append(log_entry)
                with open(log_path, 'w', encoding='utf-8') as f:
                    json.dump(logs, f, ensure_ascii=False, indent=2)
                # --- Конец записи в лог ---
                prog = load_train_progress(session_dir)
                prog['status'] = 'done'
                prog['metrics'] = metrics
                update_train_progress(prog, session_dir)
            except Exception as e:
                logger.error(f"Ошибка обучения: {e}")
                update_train_progress({'status': 'error', 'log': f'Ошибка обучения: {str(e)}'}, session_dir)
        train_thread = threading.Thread(target=train_job, args=(model_type, pretty_model_type))
        train_thread.start()
        return jsonify({'success': True})
    except Exception as e:
        logger.error(f"Ошибка обучения: {e}")
        return jsonify({'error': f'Ошибка обучения: {str(e)}'})

@app.route('/api/train_progress')
def api_train_progress():
    global last_train_temp_dir
    if not last_train_temp_dir:
        return jsonify({'status': 'not_started', 'message': 'Обучение ещё не запускалось'}), 200
    return jsonify(to_py(load_train_progress(last_train_temp_dir)))

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

@app.route('/api/model_info')
def api_model_info():
    model_file = request.args.get('model_file')
    log_path = os.path.join('model', 'model_train_log.json')
    model_type = None
    if model_file and os.path.exists(log_path):
        with open(log_path, 'r', encoding='utf-8') as f:
            logs = json.load(f)
        if isinstance(logs, dict):
            logs = [logs]
        for entry in logs:
            if entry.get('model_file') == model_file:
                model_type = entry.get('model_type')
                break
    return jsonify({'model_type': model_type or 'mobilenet_v3_small'})

def prettify_model_type(model_type):
    mapping = {
        'mobilenet_v3_small': 'MobileNetV3 Small',
        'efficientnet_b0': 'EfficientNet-B0',
        'resnet18': 'ResNet 18',
        'mobilenet_v2': 'MobileNetV2',
    }
    return mapping.get(model_type, model_type.replace('_', ' '))

if __name__ == '__main__':
    app.run(debug=True, threaded=False) 
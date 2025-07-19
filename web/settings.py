from flask import request, render_template, redirect, url_for, jsonify
import os
import json

SETTINGS_FILE = 'settings.json'

def handle_settings_request():
    model_dir = 'model'
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.pt')]
    # --- Метрики для выпадающего списка ---
    model_metrics = {}
    
    # Загружаем данные из model_train_log.json
    log_path = os.path.join(model_dir, 'model_train_log.json')
    if os.path.exists(log_path):
        try:
            with open(log_path, 'r', encoding='utf-8') as f:
                logs = json.load(f)
            if isinstance(logs, dict):
                logs = [logs]
            for entry in logs:
                name = os.path.basename(entry.get('model_file', ''))
                if name:
                    # Копируем все метрики, а не только часть
                    model_metrics[name] = entry.copy()
        except Exception:
            pass
    
    # Дополнительно загружаем данные из _stats.json файлов для каждой модели
    for model_file in model_files:
        stats_path = os.path.join(model_dir, f'{model_file}_stats.json')
        if os.path.exists(stats_path):
            try:
                with open(stats_path, 'r', encoding='utf-8') as f:
                    stats = json.load(f)
                    # Если есть вложенный metrics — разворачиваем его на верхний уровень
                    if 'metrics' in stats and isinstance(stats['metrics'], dict):
                        for k, v in stats['metrics'].items():
                            stats[k] = v
                    # confusion_matrix: приоритет metrics.confusion_matrix > confusion_matrix > None
                    if 'metrics' in stats and 'confusion_matrix' in stats['metrics']:
                        stats['confusion_matrix'] = stats['metrics']['confusion_matrix']
                    elif 'confusion_matrix' not in stats:
                        stats['confusion_matrix'] = None
                    model_metrics[model_file] = stats
            except Exception:
                pass
    settings = load_settings()
    selected_model = settings.get('model_file') if settings else None
    selected_stats = None
    message = error = None
    # Удаление модели или кэша
    if request.method == 'POST' and request.form.get('action') == 'delete_model':
        model_file = request.form.get('model_file')
        if model_file:
            try:
                os.remove(os.path.join(model_dir, model_file))
                stats_file = os.path.join(model_dir, model_file.replace('.pt', '.pt_stats.json'))
                if os.path.exists(stats_file):
                    os.remove(stats_file)
                message = f'Модель {model_file} удалена.'
                model_files = [f for f in os.listdir(model_dir) if f.endswith('.pt')]
            except Exception as e:
                error = f'Ошибка удаления: {e}'
    elif request.method == 'POST' and request.form.get('action') == 'clear_cache':
        try:
            for f in os.listdir(model_dir):
                if f.endswith('.pt_stats.json'):
                    os.remove(os.path.join(model_dir, f))
            message = 'Кэш удален.'
        except Exception as e:
            error = f'Ошибка очистки кэша: {e}'
    # Сохранение настроек
    elif request.method == 'POST':
        try:
            settings = {
                'model_file': request.form.get('model_file'),
                'low_confidence': float(request.form.get('low_confidence', 0.3)),
                'high_confidence': float(request.form.get('high_confidence', 0.7)),
                'inference_mode': request.form.get('inference_mode', 'model'),
                'manual_review_enabled': 'manual_review_enabled' in request.form,
                'export_format': request.form.get('export_format', 'standard'),
                'model_type': request.form.get('model_type', 'mobilenet_v3_small'),
                'use_optimization': 'use_optimization' in request.form
            }
            save_settings(settings)
            message = 'Настройки сохранены!'
        except Exception as e:
            error = f'Ошибка сохранения: {e}'
    # GET-запрос — просто отрисовать форму с текущими настройками
    if selected_model:
        stats_path = os.path.join(model_dir, f'{selected_model}_stats.json')
        if os.path.exists(stats_path):
            with open(stats_path, 'r', encoding='utf-8') as f:
                selected_stats = json.load(f)
                # Метрики находятся в model_stats.metrics, а не в корне
                if 'metrics' in selected_stats:
                    selected_stats.update(selected_stats['metrics'])
    return render_template('settings.html', settings=settings, model_files=model_files, model_metrics=model_metrics, model_stats=selected_stats, message=message, error=error)

def save_settings(settings):
    with open(SETTINGS_FILE, 'w', encoding='utf-8') as f:
        json.dump(settings, f, ensure_ascii=False, indent=2)

def load_settings():
    try:
        with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {
            'model_file': 'model_clashmark.pt',
            'low_confidence': 0.3,
            'high_confidence': 0.7,
            'inference_mode': 'model',
            'manual_review_enabled': True,
            'export_format': 'standard',
            'model_type': 'mobilenet_v3_small',
            'use_optimization': True
        }

@staticmethod
def get_model_stats():
    model = request.args.get('model')
    if not model:
        return jsonify({'success': False, 'error': 'Не указано имя модели'})
    stats_path = os.path.join('model', f'{model}_stats.json')
    if not os.path.exists(stats_path):
        return jsonify({'success': False, 'error': 'Файл метрик не найден'})
    with open(stats_path, 'r', encoding='utf-8') as f:
        stats = json.load(f)
    return jsonify({'success': True, 'stats': stats}) 
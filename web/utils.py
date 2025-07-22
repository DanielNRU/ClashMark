import re
import json
import os
import shutil
import zipfile

def save_uploaded_files(files, dest_dir):
    # Сохраняет загруженные файлы в указанную папку
    saved_paths = []
    for file in files:
        if file and hasattr(file, 'filename') and file.filename:
            filename = file.filename
            path = os.path.join(dest_dir, filename)
            file.save(path)
            saved_paths.append(path)
    return saved_paths

def allowed_file(filename, allowed_exts=None):
    if allowed_exts is None:
        allowed_exts = {'xml', 'zip', 'png', 'jpg'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_exts

def get_file_extension(filename):
    return os.path.splitext(filename)[1].lower()

def cleanup_temp_dir(path):
    # Удаляет временную папку/файлы
    try:
        shutil.rmtree(path, ignore_errors=True)
    except Exception:
        pass

def unzip_archives(zip_paths, dest_dir):
    # Распаковывает все zip-архивы в указанную папку
    for zip_path in zip_paths:
        if zipfile.is_zipfile(zip_path):
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(dest_dir)

def safe_filename_with_cyrillic(filename):
    """
    Очищает имя файла, оставляя только латиницу, кириллицу, цифры, точки, пробелы и дефис (дефис в конце диапазона).
    """
    filename = filename.strip()
    return re.sub(r'[^\u0000-\u007f\w. \u0430-\u044f\u0410-\u042f\u0451\u0401-]', '', filename)

def load_settings():
    try:
        with open('settings.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {
            'model_file': 'model_clashmark.pt',
            'low_confidence': 0.3,
            'high_confidence': 0.7,
            'inference_mode': 'model',
            'manual_review_enabled': True,
            'export_format': 'standard'
        }

def save_settings(settings):
    with open('settings.json', 'w', encoding='utf-8') as f:
        json.dump(settings, f, ensure_ascii=False, indent=2) 
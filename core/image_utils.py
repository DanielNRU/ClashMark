import os
import unicodedata
import logging
import hashlib
import pickle
from pathlib import Path

def normalize_filename(name):
    name = name.strip().lower()
    name = unicodedata.normalize('NFKC', name)
    return name

def normalize_path(path):
    parts = os.path.normpath(path).split(os.sep)
    return os.sep.join(normalize_filename(p) for p in parts)

def get_relative_image_path(image_href):
    """
    Извлекает относительный путь к изображению вида BSImages/ИмяФайла из абсолютного или относительного пути из XML.
    """
    if not image_href:
        return ''
    clean_path = image_href.replace('\\', '/').strip()
    parts = clean_path.split('/')
    if len(parts) >= 2:
        return os.path.join(parts[-2], parts[-1])
    elif len(parts) == 1:
        return parts[0]
    return ''

def get_absolute_image_path_optimized(image_href, session_dir):
    """
    Оптимизированная функция для получения абсолютного пути к изображению.
    Формирует путь напрямую из session_dir и относительного пути из XML без перебора файлов.
    """
    if not image_href or not session_dir:
        return None
    
    # Нормализуем путь
    rel_path = image_href.replace('\\', '/').strip()
    
    # Ищем BSImages в пути
    bsimages_idx = rel_path.find('/BSImages/')
    if bsimages_idx == -1:
        bsimages_idx = rel_path.find('BSImages/')
    
    if bsimages_idx != -1:
        # Извлекаем часть пути начиная с BSImages
        rel_path = rel_path[bsimages_idx:].lstrip('/')
        abs_path = os.path.join(session_dir, rel_path)
        
        # Проверяем существование файла
        if os.path.exists(abs_path):
            return abs_path
    
    # Fallback: если не нашли BSImages или файл не существует, 
    # извлекаем только имя файла и ищем по нему
    filename = os.path.basename(rel_path)
    if filename:
        # Ищем в папке BSImages
        bsimages_path = os.path.join(session_dir, 'BSImages', filename)
        if os.path.exists(bsimages_path):
            return bsimages_path
        
        # Ищем в корне session_dir
        root_path = os.path.join(session_dir, filename)
        if os.path.exists(root_path):
            return root_path
    
    return None


def find_image_by_name(image_href, session_dir):
    """
    Сначала ищет по относительному пути (session_dir + BSImages/ИмяФайла),
    если не найдено — ищет по имени файла во всех подпапках session_dir.
    """
    import logging
    if not image_href or not session_dir:
        logging.warning(f"[find_image_by_name] Нет image_href или session_dir: {image_href}, {session_dir}")
        return None
    rel_path = get_relative_image_path(image_href)
    abs_path = os.path.join(session_dir, rel_path)
    if os.path.exists(abs_path):
        logging.debug(f"[find_image_by_name] Найдено по относительному пути: {abs_path}")
        return abs_path
    # Fallback: ищем по имени файла во всех подпапках
    filename = os.path.basename(rel_path)
    filename_norm = normalize_filename(filename)
    for root, dirs, files in os.walk(session_dir):
        for file in files:
            if normalize_filename(file) == filename_norm:
                candidate = os.path.join(root, file)
                logging.debug(f"[find_image_by_name] Найдено по имени: {candidate}")
                return candidate
    logging.warning(f"[find_image_by_name] Не найдено: {rel_path} (session_dir={session_dir})")
    return None 

def build_image_index(session_dir, exts=(".jpg", ".jpeg", ".png", ".bmp")):
    index = {}
    for root, dirs, files in os.walk(session_dir):
        for file in files:
            if file.lower().endswith(exts):
                index[file.lower()] = os.path.join(root, file)
    return index

def get_dir_hash(directory):
    hash_md5 = hashlib.md5()
    for root, dirs, files in os.walk(directory):
        for file in sorted(files):
            path = os.path.join(root, file)
            try:
                stat = os.stat(path)
                hash_md5.update(file.encode())
                hash_md5.update(str(stat.st_size).encode())
            except Exception:
                continue
    return hash_md5.hexdigest()

def build_image_index_with_cache(session_dir, exts=(".jpg", ".jpeg", ".png", ".bmp"), cache_dir=".image_index_cache"):
    os.makedirs(cache_dir, exist_ok=True)
    dir_hash = get_dir_hash(session_dir)
    cache_path = os.path.join(cache_dir, f"index_{dir_hash}.pkl")
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except Exception:
            pass
    index = build_image_index(session_dir, exts=exts)
    try:
        with open(cache_path, 'wb') as f:
            pickle.dump(index, f)
    except Exception:
        pass
    return index

def find_image_by_name_optimized(image_href, image_index):
    if not image_href or not image_index:
        return None
    filename = os.path.basename(get_relative_image_path(image_href)).lower()
    return image_index.get(filename)

def resolve_images_vectorized_series(image_hrefs, image_index, images_dir, parallel=True):
    """
    Векторизованный и параллельный поиск изображений по href с использованием индекса.
    Если parallel=True, поиск выполняется в несколько потоков.
    """
    import pandas as pd
    from concurrent.futures import ThreadPoolExecutor
    def resolve_one(href):
        if not href:
            return ''
        rel_path = get_relative_image_path(href)
        abs_path = os.path.join(images_dir, rel_path)
        if abs_path and os.path.exists(abs_path):
            return abs_path
        image_name = os.path.basename(href)
        return image_index.get(image_name.lower(), '')
    if parallel:
        with ThreadPoolExecutor() as executor:
            return pd.Series(list(executor.map(resolve_one, image_hrefs)), index=image_hrefs.index)
    else:
        return image_hrefs.map(resolve_one) 
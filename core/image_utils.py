import os
import unicodedata
import logging
import hashlib
import pickle
from pathlib import Path
import pandas as pd

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
    try:
        if pd.isna(image_href):
            return ''
        image_href = str(image_href)
        clean_path = image_href.replace('\\', '/').strip()
        parts = clean_path.split('/')
        # Если путь содержит BSImages, берем всё после него
        for i, part in enumerate(parts):
            if part.lower() == 'bsimages':
                return os.path.join(*parts[i:])
        # Если BSImages не найден, берем последние две части
        if len(parts) >= 2:
            return os.path.join(parts[-2], parts[-1])
        elif len(parts) == 1:
            return parts[0]
        return ''
    except Exception as e:
        logging.error(f"Error in get_relative_image_path: {str(e)}")
        return ''

def get_absolute_image_path_optimized(image_href, session_dir):
    """
    Оптимизированная функция для получения абсолютного пути к изображению.
    Формирует путь напрямую из session_dir и относительного пути из XML без перебора файлов.
    """
    if not image_href or not session_dir:
        return None
    try:
        if pd.isna(image_href):
            return None
        image_href = str(image_href)
        rel_path = get_relative_image_path(image_href)
        if not rel_path:
            return None
        paths_to_try = [
            os.path.join(session_dir, rel_path),  # Полный относительный путь
            os.path.join(session_dir, 'BSImages', os.path.basename(rel_path)),  # В папке BSImages
            os.path.join(session_dir, os.path.basename(rel_path))  # В корне session_dir
        ]
        for path in paths_to_try:
            if os.path.exists(path):
                return path
        return None
    except Exception as e:
        logging.error(f"Error in get_absolute_image_path_optimized: {str(e)}")
        return None

def find_image_by_name(image_href, session_dir):
    """
    Сначала ищет по относительному пути (session_dir + BSImages/ИмяФайла),
    если не найдено — ищет по имени файла во всех подпапках session_dir.
    """
    import logging
    logger = logging.getLogger(__name__)
    if not image_href or not session_dir:
        logger.warning(f"[find_image_by_name] Нет image_href или session_dir: {image_href}, {session_dir}")
        return None
    try:
        if pd.isna(image_href):
            return None
        image_href = str(image_href)
        # Сначала пробуем оптимизированный поиск
        optimized_result = get_absolute_image_path_optimized(image_href, session_dir)
        if optimized_result:
            return optimized_result
        # Если не нашли, ищем по имени файла во всех подпапках
        filename = os.path.basename(image_href)
        for root, dirs, files in os.walk(session_dir):
            for file in files:
                if file == filename:  # Точное совпадение
                    candidate = os.path.join(root, file)
                    logger.debug(f"[find_image_by_name] Найдено по точному имени: {candidate}")
                    return candidate
        logger.warning(f"[find_image_by_name] Не найдено: {filename} (session_dir={session_dir})")
        return None
    except Exception as e:
        logger.error(f"Error in find_image_by_name: {str(e)}")
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
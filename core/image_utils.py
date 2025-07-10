import os
import unicodedata

def extract_images_dir_from_href(image_href):
    if not image_href:
        return ''
    clean_path = image_href.replace('\\', '/').strip()
    dir_path = os.path.dirname(clean_path)
    return "uploads/" + dir_path

def find_image_by_href(image_href, images_dir):
    if not image_href:
        return None
    clean_path = image_href.replace('\\', '/').strip()
    filename = os.path.basename(clean_path)
    full_path = os.path.join(images_dir, filename)
    if os.path.exists(full_path):
        return full_path
    return None

def normalize_filename(name):
    name = name.strip().lower()
    name = unicodedata.normalize('NFKC', name)
    return name

def normalize_path(path):
    # Нормализуем путь и все его компоненты
    parts = os.path.normpath(path).split(os.sep)
    return os.sep.join(normalize_filename(p) for p in parts)

def get_relative_image_path(image_href):
    """
    Извлекает относительный путь к изображению вида BSImages/ИмяФайла из абсолютного пути из XML.
    """
    if not image_href:
        return ''
    clean_path = image_href.replace('\\', '/').replace('\\', '/').strip()
    parts = clean_path.split('/')
    if len(parts) >= 2:
        return '/'.join(parts[-2:])
    elif len(parts) == 1:
        return parts[0]
    return ''

def find_image_by_relative_path(image_href, session_dir):
    """
    Склеивает временную папку и относительный путь, возвращает путь к файлу если он существует.
    """
    rel_path = get_relative_image_path(image_href)
    if not rel_path:
        return None
    abs_path = os.path.join(session_dir, rel_path)
    if os.path.exists(abs_path):
        return abs_path
    return None

# Функции поиска изображений используются для поиска файлов для visual-пар (см. алгоритм анализа)
def find_image_by_name(image_name, images_dir):
    # Fallback-функция, если прямой путь не найден
    if not image_name or not images_dir:
        print(f'[find_image_by_name] Нет имени или директории: {image_name}, {images_dir}')
        return None
    image_name = os.path.basename(image_name)
    image_name_norm = normalize_filename(image_name)
    candidates = []
    for root, dirs, files in os.walk(images_dir):
        root_norm = normalize_path(root)
        for file in files:
            file_norm = normalize_filename(file)
            candidates.append((file_norm, os.path.join(root, file), root_norm))
    print(f'[find_image_by_name] Ищу: {image_name_norm} среди {len(candidates)} файлов')
    for file_norm, path, root_norm in candidates:
        if file_norm == image_name_norm:
            print(f'[find_image_by_name] Найдено точное совпадение: {path}')
            return path
    base_name = os.path.splitext(image_name_norm)[0]
    for file_norm, path, root_norm in candidates:
        if os.path.splitext(file_norm)[0] == base_name:
            print(f'[find_image_by_name] Найдено совпадение по base_name: {path}')
            return path
    for file_norm, path, root_norm in candidates:
        if file_norm.startswith(base_name):
            print(f'[find_image_by_name] Найдено совпадение по startswith: {path}')
            return path
    print(f'[find_image_by_name] Не найдено: {image_name_norm}')
    return None 
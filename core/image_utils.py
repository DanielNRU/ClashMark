import os
import unicodedata
import logging

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
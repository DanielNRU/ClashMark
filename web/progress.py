import os
import json

def get_train_progress_path(temp_dir):
    return os.path.join(temp_dir, 'train_progress.json')

def load_train_progress(temp_dir):
    path = get_train_progress_path(temp_dir)
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_train_progress(data, temp_dir):
    path = get_train_progress_path(temp_dir)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def update_train_progress(data, temp_dir):
    prog = load_train_progress(temp_dir)
    prog.update(data)
    save_train_progress(prog, temp_dir)
# Прогресс обучения не влияет на алгоритм анализа, но используется для обучения модели visual-пар 
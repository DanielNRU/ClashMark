import pandas as pd
import yaml

# Загружаем пары из category_pairs.yaml
visual_pairs, can_pairs, cannot_pairs = set(), set(), set()
with open('category_pairs.yaml', encoding='utf-8') as f:
    yml = yaml.safe_load(f)
    visual_pairs.update([(a, b) for a, b in yml['visual']])
    visual_pairs.update([(b, a) for a, b in yml['visual']])
    can_pairs.update([(a, b) for a, b in yml['can']])
    can_pairs.update([(b, a) for a, b in yml['can']])
    cannot_pairs.update([(a, b) for a, b in yml['cannot']])
    cannot_pairs.update([(b, a) for a, b in yml['cannot']])

# Файлы для анализа
csv_files = [
    ('ВК', 'test_training_data_final.csv'),
    ('ОВ', 'test_training_data_final_OV.csv'),
]

for label, csv_path in csv_files:
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f'Файл {csv_path} не найден или не читается: {e}')
        continue
    print(f'\nФайл: {csv_path}')
    visual_df = df[df.apply(lambda row: (row['element1_category'], row['element2_category']) in visual_pairs, axis=1)]
    print('Распределение статусов среди visual-пар:')
    print(visual_df['status'].value_counts()) 
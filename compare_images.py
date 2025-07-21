import xml.etree.ElementTree as ET
from pathlib import Path

xml_path = 'Обучение/ВК со всеми.xml'
images_dir = Path('Обучение/BSImages')

# Получаем список файлов в папке
actual_files = set(f.name for f in images_dir.iterdir() if f.is_file())

# Собираем имена файлов, которые нужны по XML
needed_files = set()
tree = ET.parse(xml_path)
root = tree.getroot()
for clash in root.iter('imagePath'):
    if clash.text:
        needed_files.add(Path(clash.text.strip()).name)

missing = needed_files - actual_files

# Удалить все print(...). 
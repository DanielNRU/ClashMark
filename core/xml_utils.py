import pandas as pd
import xml.etree.ElementTree as ET
import os
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

def load_all_category_pairs(yaml_file='category_pairs.yaml'):
    """Загружает все категории пар из YAML файла"""
    import yaml
    try:
        with open(yaml_file, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
            return data
    except Exception as e:
        logger.error(f"Ошибка загрузки {yaml_file}: {e}")
        return {'can': [], 'cannot': [], 'visual': []}

def get_pair(row):
    """Получает пару категорий из строки DataFrame"""
    cat1 = str(row.get('element1_category', '')).strip()
    cat2 = str(row.get('element2_category', '')).strip()
    return (cat1, cat2) if cat1 <= cat2 else (cat2, cat1)

# get_pair и парсинг XML используются для определения класса (can/cannot/visual) согласно алгоритму анализа
def parse_xml_data(xml_path, export_format='standard'):
    """
    Парсит XML-файл и возвращает DataFrame с коллизиями (универсально для bimstep и стандартного формата)
    """
    import xml.etree.ElementTree as ET
    import pandas as pd

    tree = ET.parse(xml_path)
    root = tree.getroot()
    clashes = []
    if export_format == 'bimstep':
        for clash_result in root.findall('.//BimStepXMlClash'):
            clash = {}
            number_clash_nwc = clash_result.find('.//numberClashNwc')
            clash['clash_id'] = number_clash_nwc.text if number_clash_nwc is not None else ''
            clash['clash_name'] = clash['clash_id']
            status_elem = clash_result.find('.//status')
            clash['status'] = status_elem.text if status_elem is not None else ''
            distance_elem = clash_result.find('.//distance')
            clash['distance'] = float(distance_elem.text.replace(',', '.')) if distance_elem is not None and distance_elem.text else 0
            image_path_elem = clash_result.find('.//imagePath')
            image_path = image_path_elem.text if image_path_elem is not None else ''
            clash['image_href'] = image_path.replace('\\', '/').strip() if image_path else ''
            point_elem = clash_result.find('.//point')
            point_text = point_elem.text if point_elem is not None else '0;0;0'
            point_parts = point_text.split(';') if point_text else ['0', '0', '0']
            clash['clash_x'] = float(point_parts[0].replace(',', '.')) if len(point_parts) > 0 and point_parts[0] else 0
            clash['clash_y'] = float(point_parts[1].replace(',', '.')) if len(point_parts) > 1 and point_parts[1] else 0
            clash['clash_z'] = float(point_parts[2].replace(',', '.')) if len(point_parts) > 2 and point_parts[2] else 0
            clash['grid_location'] = ''
            status = clash['status'].lower() if clash['status'] else ''
            if status == 'active':
                clash['resultstatus'] = 'Активн.'
                clash['IsResolved'] = 1
            elif status == 'approved':
                clash['resultstatus'] = 'Подтверждено'
                clash['IsResolved'] = 0
            elif status == 'resolved':
                clash['resultstatus'] = 'Разрешено'
                clash['IsResolved'] = 0
            elif status == 'reviewed':
                clash['resultstatus'] = 'Проанализировано'
                clash['IsResolved'] = -1
            else:
                clash['resultstatus'] = 'Новый'
                clash['IsResolved'] = -1
            element_cats = clash_result.findall('.//ElementCats/string')
            for i, prefix in zip([0, 1], ['element1', 'element2']):
                clash[f'{prefix}_category'] = ''
                clash[f'{prefix}_family'] = ''
                clash[f'{prefix}_type'] = ''
                if len(element_cats) > i and element_cats[i] is not None and element_cats[i].text is not None:
                    clash[f'{prefix}_category'] = str(element_cats[i].text)
            # --- Новый блок: парсим ElementIds ---
            element_ids = clash_result.find('.//ElementIds')
            element1_id = ''
            element2_id = ''
            if element_ids is not None:
                ids = element_ids.findall('string')
                if len(ids) > 0 and ids[0] is not None and ids[0].text is not None:
                    element1_id = ids[0].text.strip()
                if len(ids) > 1 and ids[1] is not None and ids[1].text is not None:
                    element2_id = ids[1].text.strip()
            clash['element1_id'] = element1_id
            clash['element2_id'] = element2_id
            # --- Генерируем clash_uid ---
            clash['clash_uid'] = f"{min(element1_id, element2_id)}_{max(element1_id, element2_id)}" if element1_id and element2_id else ''
            clashes.append(clash)
    else:
        for clash_result in root.findall('.//clashresult'):
            clash = {}
            clash['clash_id'] = clash_result.get('guid', '')
            clash['clash_name'] = clash_result.get('name', '')
            clash['status'] = clash_result.get('status', '')
            clash['distance'] = float(clash_result.get('distance', 0))
            image_href = clash_result.get('href', '')
            clash['image_href'] = image_href.replace('\\', '/').strip() if image_href else ''
            cp = clash_result.find('.//clashpoint/pos3f')
            clash['clash_x'] = float(cp.get('x', 0)) if cp is not None and cp.get('x', None) is not None else 0
            clash['clash_y'] = float(cp.get('y', 0)) if cp is not None and cp.get('y', None) is not None else 0
            clash['clash_z'] = float(cp.get('z', 0)) if cp is not None and cp.get('z', None) is not None else 0
            grid = clash_result.find('.//gridlocation')
            clash['grid_location'] = grid.text if grid is not None and grid.text else ''
            rs = clash_result.find('.//resultstatus')
            clash['resultstatus'] = rs.text.strip() if rs is not None and rs.text else ''
            if clash['resultstatus'] == 'Активн.':
                clash['IsResolved'] = 1
            elif clash['resultstatus'] == 'Подтверждено':
                clash['IsResolved'] = 0
            else:
                clash['IsResolved'] = -1
            objs = clash_result.findall('.//clashobject')
            for i, prefix in zip([0, 1], ['element1', 'element2']):
                if len(objs) > i and objs[i] is not None:
                    obj = objs[i]
                    path_nodes = obj.findall('.//pathlink/node')
                    path_parts = [n.text for n in path_nodes if n is not None and n.text]
                    clash[f'{prefix}_category'] = path_parts[3] if len(path_parts) > 3 else ''
                    clash[f'{prefix}_family'] = path_parts[4] if len(path_parts) > 4 else ''
                    clash[f'{prefix}_type'] = path_parts[5] if len(path_parts) > 5 else ''
            # --- Новый блок: парсим ElementIds ---
            element1_id = ''
            element2_id = ''
            for i, prefix in zip([0, 1], ['element1', 'element2']):
                if len(objs) > i and objs[i] is not None:
                    obj = objs[i]
                    # id может быть в атрибуте id или в подэлементе
                    eid = obj.get('id', '')
                    if not eid:
                        id_elem = obj.find('.//id')
                        if id_elem is not None and id_elem.text is not None:
                            eid = id_elem.text.strip()
                    if prefix == 'element1':
                        element1_id = eid
                    else:
                        element2_id = eid
            clash['element1_id'] = element1_id
            clash['element2_id'] = element2_id
            # --- Генерируем clash_uid ---
            clash['clash_uid'] = f"{min(element1_id, element2_id)}_{max(element1_id, element2_id)}" if element1_id and element2_id else ''
            clashes.append(clash)
    df = pd.DataFrame(clashes)
    return df

def export_to_xml(df, output_path, original_xml_path=None):
    """Экспортирует DataFrame в XML формат"""
    try:
        # Создаем корневой элемент
        root = ET.Element('ClashResults')
        
        for _, row in df.iterrows():
            clash = ET.SubElement(root, 'Clash')
            # --- clash_uid как id ---
            clash.set('id', str(row.get('clash_uid', row.get('clash_id', ''))))
            clash.set('name', str(row.get('clash_name', '')))
            
            # Element1
            element1 = ET.SubElement(clash, 'Element1')
            element1.set('id', str(row.get('element1_id', '')))
            element1.set('category', str(row.get('element1_category', '')))
            element1.set('family', str(row.get('element1_family', '')))
            element1.set('type', str(row.get('element1_type', '')))
            
            # Element2
            element2 = ET.SubElement(clash, 'Element2')
            element2.set('id', str(row.get('element2_id', '')))
            element2.set('category', str(row.get('element2_category', '')))
            element2.set('family', str(row.get('element2_family', '')))
            element2.set('type', str(row.get('element2_type', '')))
            
            # Point
            point = ET.SubElement(clash, 'Point')
            point.set('x', str(row.get('clash_x', 0)))
            point.set('y', str(row.get('clash_y', 0)))
            point.set('z', str(row.get('clash_z', 0)))
            
            # Distance
            distance = ET.SubElement(clash, 'Distance')
            distance.text = str(row.get('distance', 0))
            
            # Status
            status = ET.SubElement(clash, 'Status')
            pred = row.get('cv_prediction')
            pred_source = row.get('prediction_source', '')
            if pred == 1:
                status.text = 'Активн.'
            elif pred == 0:
                status.text = 'Подтверждено'
            elif pred == -1 and pred_source == 'manual_review':
                status.text = 'Проанализировано'
            elif pred == -1 and pred_source == 'algorithm':
                status.text = 'Требует проверки'
            else:
                status.text = 'Новый'
            
            # Confidence
            confidence = ET.SubElement(clash, 'Confidence')
            confidence.text = str(row.get('cv_confidence', 0))
            
            # Image
            if row.get('image_href'):
                image = ET.SubElement(clash, 'Image')
                image.set('href', str(row.get('image_href', '')))
        
        # Создаем дерево и сохраняем
        tree = ET.ElementTree(root)
        tree.write(output_path, encoding='utf-8', xml_declaration=True)
        
        logger.info(f"Экспортировано {len(df)} коллизий в {output_path}")
        
    except Exception as e:
        logger.error(f"Ошибка экспорта в XML {output_path}: {e}")
        raise

def export_to_bimstep_xml(df, output_xml_path, original_xml_path=None):
    """Экспортирует результаты в BIM Step формат"""
    try:
        # Регистрируем пространства имен
        ET.register_namespace('xsd', 'http://www.w3.org/2001/XMLSchema')
        ET.register_namespace('xsi', 'http://www.w3.org/2001/XMLSchema-instance')
        
        # Создаем корневой элемент
        root = ET.Element('BimStepXML')
        
        # Создаем список коллизий
        list_clashes = ET.SubElement(root, 'ListBimStepXMlClash')
        
        for _, row in df.iterrows():
            clash = ET.SubElement(list_clashes, 'BimStepXMlClash')
            
            # ElementIds
            element_ids = ET.SubElement(clash, 'ElementIds')
            ET.SubElement(element_ids, 'string').text = str(row.get('element1_id', ''))
            ET.SubElement(element_ids, 'string').text = str(row.get('element2_id', ''))
            
            # ElementCats
            element_cats = ET.SubElement(clash, 'ElementCats')
            ET.SubElement(element_cats, 'string').text = str(row.get('element1_category', ''))
            ET.SubElement(element_cats, 'string').text = str(row.get('element2_category', ''))
            
            # Point - используем правильный формат с точкой с запятой
            point = ET.SubElement(clash, 'point')
            point.text = f"{row.get('clash_x', 0)};{row.get('clash_y', 0)};{row.get('clash_z', 0)}"
            
            # Status - правильное сопоставление
            status = ET.SubElement(clash, 'status')
            pred = row.get('cv_prediction')
            pred_source = row.get('prediction_source', '')
            if pred == 1:
                status.text = 'Active'  # cannot -> active
            elif pred == 0:
                status.text = 'Approved'  # can -> approved
            elif pred == -1 and pred_source == 'manual_review':
                status.text = 'Reviewed'  # visual -> reviewed (ручная разметка)
            elif pred == -1 and pred_source == 'algorithm':
                status.text = 'Reviewed'  # visual -> reviewed (алгоритм)
            else:
                status.text = 'New'  # всё остальное
            
            # NameFile
            name_file = ET.SubElement(clash, 'nameFile')
            name_file.text = os.path.splitext(os.path.basename(original_xml_path))[0] if original_xml_path else 'ClashMark'
            
            # Distance
            distance = ET.SubElement(clash, 'distance')
            distance.text = str(row.get('distance', 0))
            
            # Tolerance
            tolerance = ET.SubElement(clash, 'tolerance')
            tolerance.text = '0,00328083989501312'
            
            # NumberClashNwc (оставим для совместимости)
            number_clash = ET.SubElement(clash, 'numberClashNwc')
            number_clash.text = str(row.get('clash_name', row.get('clash_uid', row.get('clash_id', ''))))
            
            # ImagePath - сохраняем исходные пути к изображениям
            image_path = ET.SubElement(clash, 'imagePath')
            if row.get('image_href'):
                # Сохраняем исходный путь к изображению без изменений
                image_path.text = row.get('image_href', '')
            else:
                image_path.text = ''
            
            # Models
            models = ET.SubElement(clash, 'Models')
            ET.SubElement(models, 'string').text = str(row.get('element1_family', ''))
            ET.SubElement(models, 'string').text = str(row.get('element2_family', ''))
        
        # Отладочная информация
        status_counts = {}
        for _, row in df.iterrows():
            pred = row.get('cv_prediction')
            if pred == 0:
                status_counts['Approved'] = status_counts.get('Approved', 0) + 1
            elif pred == 1:
                status_counts['Active'] = status_counts.get('Active', 0) + 1
            else:
                status_counts['Reviewed'] = status_counts.get('Reviewed', 0) + 1
        
        logger.info(f"Статусы для экспорта: {status_counts}")
        
        # Создаем красивый XML с отступами
        def indent(elem, level=0):
            i = "\n" + level*"  "
            if len(elem):
                if not elem.text or not elem.text.strip():
                    elem.text = i + "  "
                if not elem.tail or not elem.tail.strip():
                    elem.tail = i
                for subelem in elem:
                    indent(subelem, level+1)
                if not elem.tail or not elem.tail.strip():
                    elem.tail = i
            else:
                if level and (not elem.tail or not elem.tail.strip()):
                    elem.tail = i
        
        indent(root)
        
        # Сохраняем с правильной кодировкой и объявлением XML
        xml_str = ET.tostring(root, encoding='unicode')
        xml_str = xml_str.replace(
            '<BimStepXML>',
            '<BimStepXML xmlns:xsd="http://www.w3.org/2001/XMLSchema" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
        )
        
        with open(output_xml_path, 'w', encoding='utf-8') as f:
            f.write("<?xml version='1.0' encoding='UTF-8'?>\n")
            f.write(xml_str)
        
        logger.info(f"Экспортировано {len(df)} коллизий в BIM Step формат: {output_xml_path}")
        
    except Exception as e:
        logger.error(f"Ошибка экспорта в BIM Step XML {output_xml_path}: {e}")
        raise

def add_bimstep_journal_entry(clash_uid, prediction_type, comment, session_dir=None, element1_id=None, element2_id=None, status=None):
    """Добавляет запись в журнал BIM Step с ID объектов и статусом"""
    try:
        if session_dir:
            # Создаем журнал во временной папке сессии
            journal_path = os.path.join(session_dir, 'JournalBimStep.xml')
        else:
            # Fallback на папку data
            journal_path = 'data/JournalBimStep.xml'
            os.makedirs('data', exist_ok=True)
        
        # Создаем файл журнала, если его нет
        if not os.path.exists(journal_path):
            # Создаем XML с правильными пространствами имен
            xml_content = '''<?xml version="1.0"?>
<WorkSharingInfos xmlns:xsd="http://www.w3.org/2001/XMLSchema" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <ListWorkSharingInfo>
  </ListWorkSharingInfo>
</WorkSharingInfos>'''
            with open(journal_path, 'w', encoding='utf-8') as f:
                f.write(xml_content)
        
        # Читаем существующий журнал
        tree = ET.parse(journal_path)
        root = tree.getroot()
        
        # Находим или создаем ListWorkSharingInfo
        list_work_sharing_info = root.find('ListWorkSharingInfo')
        if list_work_sharing_info is None:
            list_work_sharing_info = ET.SubElement(root, 'ListWorkSharingInfo')
        
        # Создаем новую запись WorkSharingInfo
        work_sharing_info = ET.SubElement(list_work_sharing_info, 'WorkSharingInfo')
        
        # Добавляем ID объектов
        id1_elem = ET.SubElement(work_sharing_info, 'id1')
        id1_elem.text = str(element1_id) if element1_id else ''
        
        id2_elem = ET.SubElement(work_sharing_info, 'id2')
        id2_elem.text = str(element2_id) if element2_id else ''
        
        # Создаем ListStory
        list_story = ET.SubElement(work_sharing_info, 'ListStory')
        
        # Создаем Story
        story = ET.SubElement(list_story, 'Story')
        
        # Добавляем время
        data_elem = ET.SubElement(story, 'data')
        data_elem.text = datetime.now().strftime('%d.%m.%Y %H:%M:%S')
        
        # Добавляем пользователя
        nameuser_elem = ET.SubElement(story, 'nameuser')
        nameuser_elem.text = 'ClashMark'
        
        # Определяем, является ли коллизия допустимой
        is_acceptable = (status == 'approved')
        
        # Формируем комментарий в зависимости от типа разметки
        if prediction_type == 'manual':
            story_comment = 'Размечено с помощью ручной разметки'
        elif prediction_type == 'model':
            story_comment = 'Размечено с помощью модели компьютерного зрения'
        else:
            story_comment = 'Размечено с помощью алгоритма'
        
        # Добавляем комментарий в Story
        comment_elem = ET.SubElement(story, 'comment')
        comment_elem.text = story_comment
        
        # Добавляем тип истории
        type_story_elem = ET.SubElement(story, 'typeStory')
        if is_acceptable:
            type_story_elem.text = 'acceptable'
        else:
            type_story_elem.text = 'comment'
        
        # Добавляем ListImages (пустой)
        list_images = ET.SubElement(story, 'ListImages')
        
        # Добавляем блок acceptable только для допустимых коллизий
        if is_acceptable:
            acceptable_elem = ET.SubElement(work_sharing_info, 'acceptable')
            acc_data = ET.SubElement(acceptable_elem, 'data')
            acc_data.text = datetime.now().strftime('%d.%m.%Y %H:%M:%S')
            acc_nameuser = ET.SubElement(acceptable_elem, 'nameuser')
            acc_nameuser.text = 'ClashMark' 
            acc_comment = ET.SubElement(acceptable_elem, 'comment')
            acc_comment.text = 'Коллизия разрешена'
        
        # Сохраняем журнал
        tree.write(journal_path, encoding='utf-8', xml_declaration=True)
        
        logger.info(f"Добавлена запись в журнал BIM Step для коллизии {clash_uid}")
        
    except Exception as e:
        logger.error(f"Ошибка добавления записи в журнал BIM Step: {e}")
        raise 
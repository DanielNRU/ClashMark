from flask import request, render_template, current_app, send_file
import os
import tempfile
import shutil
import traceback
from web.utils import save_uploaded_files, cleanup_temp_dir, unzip_archives, safe_filename_with_cyrillic, load_settings
from ml.inference import collect_dataset_from_multiple_files, predict
from ml.model import create_model
from ml.dataset import create_transforms
from core.xml_utils import export_to_xml, export_to_bimstep_xml
from core.image_utils import find_image_by_name

def handle_inference_request():
    if request.method == 'POST':
        temp_dir = None
        try:
            # Сохраняем загруженные XML-файлы
            xml_files = request.files.getlist('xml_file')
            zip_files = request.files.getlist('zip_file')
            temp_dir = tempfile.mkdtemp()
            settings = load_settings()
            export_format = settings.get('export_format', 'standard')
            
            # Сохраняем XML файлы
            xml_paths = []
            for xml_file in xml_files:
                if not xml_file.filename:
                    continue
                safe_name = safe_filename_with_cyrillic(xml_file.filename)
                xml_path = os.path.join(temp_dir, safe_name)
                xml_file.save(xml_path)
                xml_paths.append(xml_path)
            
            # Сохраняем и распаковываем ZIP файлы
            zip_paths = []
            for zip_file in zip_files:
                if not zip_file.filename:
                    continue
                safe_name = safe_filename_with_cyrillic(zip_file.filename)
                zip_path = os.path.join(temp_dir, safe_name)
                zip_file.save(zip_path)
                zip_paths.append(zip_path)
            
            # Распаковываем архивы
            images_dir = os.path.join(temp_dir, 'BSImages')
            os.makedirs(images_dir, exist_ok=True)
            unzip_archives(zip_paths, images_dir)
            unzip_archives(zip_paths, temp_dir)
            
            # Собираем датасет (сначала BSImages, потом temp_dir)
            df = collect_dataset_from_multiple_files(xml_paths, images_dir=images_dir, export_format=export_format)
            if len(df) == 0:
                df = collect_dataset_from_multiple_files(xml_paths, images_dir=temp_dir, export_format=export_format)
            
            if len(df) == 0:
                cleanup_temp_dir(temp_dir)
                return render_template('index.html', error='Не найдено изображений для анализа. Проверьте, что имена файлов в XML совпадают с именами изображений в архиве, и что архив содержит папку BSImages или изображения в корне.')
            
            # Загружаем модель
            device = 'cpu'
            model = create_model(device)
            transform = create_transforms(is_training=False)
            
            # Получаем пороги уверенности из настроек
            low_confidence = settings.get('low_confidence', 0.3)
            high_confidence = settings.get('high_confidence', 0.7)
            
            df_pred = predict(model, device, df, transform, 
                           low_confidence_threshold=low_confidence, 
                           high_confidence_threshold=high_confidence)
            
            # Экспортируем результат
            output_xml = os.path.join(temp_dir, 'cv_results.xml')
            if export_format == 'bimstep':
                export_to_bimstep_xml(df_pred, output_xml, original_xml_path=xml_paths[0])
            else:
                export_to_xml(df_pred, output_xml, original_xml_path=xml_paths[0])
            
            return send_file(output_xml, as_attachment=True, download_name='cv_results.xml')
            
        except Exception as e:
            if temp_dir:
                cleanup_temp_dir(temp_dir)
            return render_template('index.html', error=f'Ошибка инференса: {e}\n{traceback.format_exc()}')
    
    return render_template('index.html')

def process_inference(xml_files, zip_files, settings):
    # Здесь будет логика инференса модели
    # Возвращать результат, логи, ошибки
    pass

def get_inference_results():
    # Здесь можно реализовать получение результатов инференса
    pass

from flask import Blueprint
main_bp = Blueprint('main_inference', __name__)

@main_bp.route('/download_result')
def download_result():
    path = request.args.get('path')
    if not path or not os.path.exists(path):
        return 'Файл не найден', 404
    return send_file(path, as_attachment=True, download_name='cv_results.xml') 
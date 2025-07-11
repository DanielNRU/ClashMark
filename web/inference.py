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
            # --- Новый блок: обработка пар категорий ---
            import pandas as pd
            from core.xml_utils import load_all_category_pairs, get_pair
            pairs = load_all_category_pairs('category_pairs.yaml')
            can_pairs = set()
            cannot_pairs = set()
            visual_pairs = set()
            if 'can' in pairs and isinstance(pairs['can'], list):
                for pair in pairs['can']:
                    if isinstance(pair, list) and len(pair) == 2:
                        a, b = pair
                        can_pairs.add((a, b) if a <= b else (b, a))
            if 'cannot' in pairs and isinstance(pairs['cannot'], list):
                for pair in pairs['cannot']:
                    if isinstance(pair, list) and len(pair) == 2:
                        a, b = pair
                        cannot_pairs.add((a, b) if a <= b else (b, a))
            if 'visual' in pairs and isinstance(pairs['visual'], list):
                for pair in pairs['visual']:
                    if isinstance(pair, list) and len(pair) == 2:
                        a, b = pair
                        visual_pairs.add((a, b) if a <= b else (b, a))
            # Получаем настройки
            inference_mode = settings.get('inference_mode', 'model')
            manual_review_enabled = settings.get('manual_review_enabled', False)
            low_confidence = settings.get('low_confidence', 0.3)
            high_confidence = settings.get('high_confidence', 0.7)
            # Классификация
            df['cv_prediction'] = None
            df['cv_confidence'] = None
            df['prediction_source'] = None
            df['cv_status'] = None
            visual_rows = []
            visual_indices = []
            for idx, row in df.iterrows():
                pair = get_pair(row)
                if pair in can_pairs:
                    df.at[idx, 'cv_prediction'] = 0
                    df.at[idx, 'cv_confidence'] = 1.0
                    df.at[idx, 'prediction_source'] = 'algorithm'
                    df.at[idx, 'cv_status'] = 'Approved'
                elif pair in cannot_pairs:
                    df.at[idx, 'cv_prediction'] = 1
                    df.at[idx, 'cv_confidence'] = 1.0
                    df.at[idx, 'prediction_source'] = 'algorithm'
                    df.at[idx, 'cv_status'] = 'Active'
                elif pair in visual_pairs:
                    df.at[idx, 'cv_prediction'] = -1
                    df.at[idx, 'cv_confidence'] = 0.5
                    df.at[idx, 'prediction_source'] = 'algorithm'
                    df.at[idx, 'cv_status'] = 'Reviewed'
                    visual_rows.append(row)
                    visual_indices.append(idx)
                else:
                    df.at[idx, 'cv_prediction'] = -1
                    df.at[idx, 'cv_confidence'] = 0.5
                    df.at[idx, 'prediction_source'] = 'algorithm'
                    df.at[idx, 'cv_status'] = 'Reviewed'
                    visual_rows.append(row)
                    visual_indices.append(idx)
            # Обработка visual
            if visual_rows:
                if inference_mode == 'model':
                    import torch
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    visual_df = pd.DataFrame(visual_rows)
                    transform = create_transforms(is_training=False)
                    model = create_model(device)
                    model_path = os.path.join('model', settings.get('model_file', 'model_clashmark.pt'))
                    model.load_state_dict(torch.load(model_path, map_location=device))
                    model.to(device)
                    model.eval()
                    from ml.inference import predict
                    visual_pred_df = predict(model, device, visual_df, transform, 
                                           low_confidence_threshold=low_confidence, 
                                           high_confidence_threshold=high_confidence)
                    for i, idx in enumerate(visual_indices):
                        prediction = int(visual_pred_df.iloc[i]['cv_prediction'])
                        confidence = float(visual_pred_df.iloc[i]['cv_confidence'])
                        if prediction == -1:
                            df.at[idx, 'cv_prediction'] = -1
                            df.at[idx, 'cv_confidence'] = confidence
                            df.at[idx, 'prediction_source'] = 'model_uncertain'
                            df.at[idx, 'cv_status'] = 'Reviewed'
                        else:
                            df.at[idx, 'cv_prediction'] = prediction
                            df.at[idx, 'cv_confidence'] = confidence
                            df.at[idx, 'prediction_source'] = 'model'
                            df.at[idx, 'cv_status'] = 'Approved' if prediction == 0 else 'Active'
                elif manual_review_enabled:
                    for idx in visual_indices:
                        df.at[idx, 'cv_prediction'] = -1
                        df.at[idx, 'cv_confidence'] = 0.5
                        df.at[idx, 'prediction_source'] = 'manual_review'
                        df.at[idx, 'cv_status'] = 'Reviewed'
                else:
                    for idx in visual_indices:
                        df.at[idx, 'cv_prediction'] = -1
                        df.at[idx, 'cv_confidence'] = 0.5
                        df.at[idx, 'prediction_source'] = 'algorithm'
                        df.at[idx, 'cv_status'] = 'Reviewed'
            # --- Конец нового блока ---
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
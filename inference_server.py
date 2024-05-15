from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import torch
from paddleocr import PaddleOCR
from transformers import AutoModelForTokenClassification, AutoProcessor
from collections import defaultdict
import json
import io

app = Flask(__name__)

# Define labels and initialize models
labels = ['I-COMPANY', 'I-DATE', 'I-ADDRESS', 'I-TOTAL', 'I-TAX', 'I-PRODUCT', 'O']
id2label = {v: k for v, k in enumerate(labels)}

from ppocr.utils.logging import get_logger
import logging
logger = get_logger()
logger.setLevel(logging.ERROR)

ocr = PaddleOCR(use_angle_cls=True, lang='sv', det_db_score_mode='slow', rec_db_score_mode='slow',
                binarize=False, use_dilation=True, use_space_char=True, det_db_unclip_ratio=1.72,
                rec_model_dir='/mnt/c/School/Exjobb/rec_ppocr_v3_distillation/best_model',
                det_model_dir='/mnt/c/School/Exjobb/det_model/content/PaddleOCR/output/ch_PP-OCR_V3_det/best_model')
processor = AutoProcessor.from_pretrained("/mnt/c/School/Exjobb/LMv3Model", apply_ocr=False)
model = AutoModelForTokenClassification.from_pretrained("/mnt/c/School/Exjobb/LMv3Model")

def process_bbox(box):
    x_coords = [point[0] for point in box]
    y_coords = [point[1] for point in box]
    x_min, y_min = min(x_coords), min(y_coords)
    x_max, y_max = max(x_coords), max(y_coords)
    return [x_min, y_min, x_max, y_max]

def normalize_bbox(bbox, size):
    width, height = size
    return [int(1000 * coord / width if i % 2 == 0 else 1000 * coord / height) for i, coord in enumerate(bbox)]

def dataSetFormat(img_file):
    ress = ocr.ocr(np.asarray(img_file))
    test_dict = {'tokens': [], 'bboxes': [], 'img_path': img_file}
    for item in ress[0]:
        test_dict['tokens'].append(item[1][0])
        test_dict['bboxes'].append(normalize_bbox(process_bbox(item[0]), img_file.size))
    return ress, test_dict

@app.route('/process', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    image_file = request.files['image']
    image = Image.open(io.BytesIO(image_file.read()))
    image.save('debug_output_image.png')  # Save the image to disk for debugging
    size = image.size
    ress, test_dict = dataSetFormat(image)

    # Process image and tokens
    encoding = processor(test_dict['img_path'].convert('RGB'), test_dict['tokens'], boxes=test_dict['bboxes'],
                         max_length=512, padding="max_length", truncation=True, return_tensors='pt',
                         return_offsets_mapping=True)

    offset_mapping = encoding.pop('offset_mapping')
    inputs_ids = torch.tensor(encoding['input_ids'], dtype=torch.int64).flatten()
    attention_mask = torch.tensor(encoding['attention_mask'], dtype=torch.int64).flatten()
    bbox = torch.tensor(encoding['bbox'], dtype=torch.int64).flatten(end_dim=1)
    pixel_values = torch.tensor(encoding['pixel_values'], dtype=torch.float32).flatten(end_dim=1)

    # Move tensors to appropriate device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for k, v in encoding.items():
        encoding[k] = v.to(device)

    outputs = model(input_ids=inputs_ids.unsqueeze(0), attention_mask=attention_mask.unsqueeze(0),
                    bbox=bbox.unsqueeze(0), pixel_values=pixel_values.unsqueeze(0))

    # Decode predictions
    predictions = outputs.logits.argmax(-1).squeeze().tolist()
    token_boxes = encoding['bbox'].squeeze().tolist()
    seen_boxes = set()
    true_predictions = []
    true_boxes = []
    is_subword = np.array(offset_mapping.squeeze().tolist())[:, 0] != 0

    for idx, (pred, box) in enumerate(zip(predictions, token_boxes)):
        if not is_subword[idx] and box != [0, 0, 0, 0]:
            normalized_box = [size[0] * (b / 1000) for b in box]
            box_tuple = tuple(normalized_box)
            if box_tuple not in seen_boxes:
                seen_boxes.add(box_tuple)
                true_predictions.append(pred)
                true_boxes.append(normalized_box)

    #print(true_predictions)

    # Filter data and format
    data = [(id2label[pred], token) for pred, token in zip(true_predictions, test_dict['tokens']) if
            id2label[pred] != 'O']
    processed_data = defaultdict(str)
    for tag, value in data:
        if tag == 'I-COMPANY':
            processed_data['company'] = (processed_data['company'] + " " + value).strip()
        if tag == 'I-ADDRESS':
            processed_data['address'] = (processed_data['address'] + " " + value).strip()
        if tag == 'I-PRODUCT' and 'product' not in processed_data:
            processed_data['products'] = value
        elif tag == 'I-PRODUCT':
            processed_data['products'] = (processed_data['products'] + " " + value).strip()
        elif tag == 'I-TOTAL' and 'total' not in processed_data:
            processed_data['total'] = value
        elif tag == 'I-TAX' and 'tax' not in processed_data:
            processed_data['tax'] = value
        elif tag == 'I-DATE' and 'date' not in processed_data:
            processed_data['date'] = value

    # Append processed data with image name
    return jsonify(dict(processed_data))


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

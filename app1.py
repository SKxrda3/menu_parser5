
from paddleocr import PaddleOCR
import os
import sys
from flask import Flask, request, jsonify
import re
import pandas as pd
from tabulate import tabulate
import mysql.connector as mysql
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from PIL import Image
load_dotenv()
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ocr = PaddleOCR(
    use_textline_orientation=True,
    lang='en',
    det_model_dir='models/det/en/en_PP-OCRv3_det_infer',
    rec_model_dir='models/rec/en/en_PP-OCRv3_rec_infer',
    cls_model_dir='models/cls/ch_ppocr_mobile_v2.0_cls_infer',

)

def preprocess_image(image_path):
    with Image.open(image_path) as img:
        img = img.convert('RGB')
        # img.thumbnail((1024, 1024), Image.ANTIALIAS)
        img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
        img.save(image_path)

mysql_config = {
    "host": os.getenv("MYSQL_HOST"),
    "user": os.getenv("MYSQL_USER"),
    "password": os.getenv("MYSQL_PASSWORD"),
    "database": os.getenv("MYSQL_DATABASE")
}

# def extract_boxes(image_path, conf_threshold=0.6):
#     results = ocr.ocr(image_path, cls=True)

#     boxes = []
#     result = results[0]

#     for box_points, text, conf in zip(result['dt_polys'], result['rec_texts'], result['rec_scores']):
#         if conf >= conf_threshold:
#             box = box_points.tolist()
#             x_center = sum([p[0] for p in box]) / len(box)
#             y_center = sum([p[1] for p in box]) / len(box)
#             boxes.append({
#                 'text': text.strip(),
#                 'conf': conf,
#                 'box': box,
#                 'x': x_center,
#                 'y': y_center
#             })
#     return boxes

def extract_boxes(image_path, conf_threshold=0.6):
    results = ocr.ocr(image_path, cls=True)
    
    boxes = []

    for line in results[0]:  # results is a list of list
        box_points = line[0]  # quadrilateral box: list of 4 [x, y] points
        text = line[1][0]     # recognized text
        conf = line[1][1]     # confidence score

        if conf >= conf_threshold:
            x_center = sum([p[0] for p in box_points]) / len(box_points)
            y_center = sum([p[1] for p in box_points]) / len(box_points)
            boxes.append({
                'text': text.strip(),
                'conf': conf,
                'box': box_points,
                'x': x_center,
                'y': y_center
            })

    return boxes


def group_by_rows(boxes, y_thresh=15):
    boxes.sort(key=lambda b: b["y"])
    rows = []
    current_row = []
    last_y = -1000

    for box in boxes:
        if abs(box["y"] - last_y) > y_thresh:
            if current_row:
                rows.append(sorted(current_row, key=lambda b: b["x"]))
            current_row = [box]
        else:
            current_row.append(box)
        last_y = box["y"]

    if current_row:
        rows.append(sorted(current_row, key=lambda b: b["x"]))
    return rows

def assign_categories(rows):
    categorized_rows = []
    current_category = "Uncategorized"

    for row in rows:
        texts = [box["text"] for box in row]
        full_line = " ".join(texts).strip()

        if detect_price(full_line):
            for box in row:
                box["category"] = current_category
            categorized_rows.append(row)
            continue

        words = full_line.split()
        uppercase_words = sum(1 for w in words if w.isupper() or w.istitle())
        is_probable_category = (
            uppercase_words >= max(1, len(words) // 2) and
            len(full_line) <= 35 and
            len(words) <= 4
        )

        if is_probable_category:
            current_category = full_line
            continue

        for box in row:
            box["category"] = current_category
        categorized_rows.append(row)

    return categorized_rows

def detect_price(text):
    return re.search(r'(\u20B9|Rs\.?)?\s?\d{1,4}([.,]\d{1,2})?', text)

def is_valid_item(text):
    if not text or len(text.strip()) <= 2:
        return False

    if text.strip().isupper() and len(text.strip()) <= 3:
        return False

    noise_keywords = { 'am', 'pm', 'yo', 'l', 't', 'a', 'b', '/', '-', '|', ':', '.', ',', '–', '—', '_', '(', ')',
                       'daily', 'only', 'each', 'per', 'day', 'week', 'month', 'with', 'served', 'includes',
                       'available', 'mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun', 'timings', 'timing',
                       'from', 'at', 'till', 'until', 'for', 'special', 'offer', 'extra', 'add-on',
                       'optional', 'combo', 'set', 'option', 'mrp', 'gst', 'inclusive', 'exclusive',
                       'taxes', 'inc.', 'excl.', 'incl.' }

    clean_text = re.sub(r'[^\w]', '', text).lower()
    if clean_text in noise_keywords:
        return False

    if not re.search(r'[a-zA-Z]', text):
        return False
    return True

def parse_rows_to_menu(categorized_rows, image_name="unknown"):
    menu = []
    last_item_entry = None

    for row in categorized_rows:
        row.sort(key=lambda b: b["x"])
        full_line = " ".join([b["text"] for b in row]).strip()
        current_category = row[0].get("category", "Uncategorized")

        price_matches = list(re.finditer(r'(₹|Rs\.?)?\s?\d{1,5}([.,]\d{1,2})?', full_line))

        if price_matches:
            items = []
            prices = []
            start = 0

            for match in price_matches:
                price_text = match.group().strip()
                price_start = match.start()
                item_chunk = full_line[start:price_start].strip(" -–—|,")
                item_texts = re.split(r'\s{2,}|,|/| - | \| |\. ', item_chunk)

                for item_text in item_texts:
                    item_text = re.sub(r'\(.*?\)', '', item_text).strip()
                    if not is_valid_item(item_text):
                        continue
                    items.append(item_text)
                    prices.append(price_text)

                start = match.end()

            for i in range(min(len(items), len(prices))):
                entry = {
                    "image": image_name,
                    "category": current_category,
                    "item": items[i],
                    "price": prices[i],
                    "description": ""
                }
                last_item_entry = entry
                menu.append(entry)

        elif last_item_entry and current_category == last_item_entry["category"]:
            last_item_entry["description"] += " " + full_line

    return menu

# def insert_into_mysql(data, host, user, password, database):
def insert_into_mysql(data, host, user, password, database, vender_id):
    connection = None
    cursor = None
    try:
        connection = mysql.connect(
            host=host,
            user=user,
            password=password,
            database=database,
            port=3306,
            use_pure=True
        )
        cursor = connection.cursor()

        insert_query = """
        INSERT INTO menu_or_services (category, item_or_service, price, description, vendor_id, image_path)
        VALUES (%s, %s, %s, %s, %s, %s)
        """

        for entry in data:
            cursor.execute(insert_query, (
                entry["category"],
                entry["item"],
                entry["price"],
                entry["description"],
                vender_id,
                entry["image"]  # Assuming you're storing the filename
            ))


        connection.commit()
        print("\n✅ Extracted menu data inserted into MySQL database.")

    except mysql.Error as err:
        print(f"❌ MySQL error: {err}")

    finally:
        if connection and connection.is_connected():
            if cursor:
                cursor.close()
            connection.close()

# def process_folder(folder_path, mysql_config):
def process_folder(folder_path, mysql_config, vender_id):
    all_data = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff")):
            image_path = os.path.join(folder_path, filename)
            print(f"Processing {filename} ...")

            boxes = extract_boxes(image_path)
            rows = group_by_rows(boxes)
            final_data = assign_categories(rows)
            menu = parse_rows_to_menu(final_data, image_name=filename)

            ignore_phrases = ["preparation time", "serving size", "cooking time", "calories"]

            for entry in menu:
                combined_text = (entry["item"] + " " + entry["category"] + " " + entry.get("description", "")).lower()
                if any(phrase in combined_text for phrase in ignore_phrases):
                    continue
                all_data.append(entry)

    if all_data:
        # insert_into_mysql(all_data, **mysql_config)
        insert_into_mysql(all_data, vender_id=vender_id, **mysql_config)
    else:
        print("No menu data extracted from images.")

# @app.route('/upload-menu', methods=['POST'])
# def upload_menu():
#     if 'image' not in request.files:
#         return jsonify({"error": "Image file is missing."}), 400
#     if 'vender_id' not in request.form:
#         return jsonify({"error": "vender_id is missing."}), 400

#     file = request.files['image']
#     vender_id = request.form['vender_id']
#     filename = secure_filename(file.filename)
#     file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#     file.save(file_path)

#     # OCR & parsing
#     boxes = extract_boxes(file_path)
#     rows = group_by_rows(boxes)
#     final_data = assign_categories(rows)
#     menu = parse_rows_to_menu(final_data, image_name=filename)

#     ignore_phrases = ["preparation time", "serving size", "cooking time", "calories"]
#     filtered_data = [
#         entry for entry in menu
#         if not any(phrase in (entry["item"] + " " + entry["category"] + " " + entry.get("description", "")).lower()
#                    for phrase in ignore_phrases)
#     ]

#     if filtered_data:
#         insert_into_mysql(filtered_data, vender_id=vender_id, **mysql_config)
#         return jsonify({"message": "Menu extracted and inserted into DB.", "items": filtered_data}), 200
#     else:
#         return jsonify({"message": "No valid menu data found in the image."}), 200

@app.route('/upload-menu', methods=['POST'])
def upload_menu():
    if 'image' not in request.files:
        return jsonify({"error": "Image file is missing."}), 400
    if 'vender_id' not in request.form:
        return jsonify({"error": "vender_id is missing."}), 400

    file = request.files['image']
    vender_id = request.form['vender_id']

    if file.filename == '':
        return jsonify({"error": "Empty filename."}), 400

    filename = secure_filename(file.filename)
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(image_path)

    preprocess_image(image_path)  # Resize for consistency

    try:
        boxes = extract_boxes(image_path)
        rows = group_by_rows(boxes)
        final_data = assign_categories(rows)
        menu = parse_rows_to_menu(final_data, image_name=filename)

        ignore_phrases = ["preparation time", "serving size", "cooking time", "calories"]
        filtered_menu = [
            entry for entry in menu
            if not any(phrase in (entry["item"] + " " + entry["category"] + " " + entry.get("description", "")).lower()
                       for phrase in ignore_phrases)
        ]

        if filtered_menu:
            insert_into_mysql(filtered_menu, vender_id=vender_id, **mysql_config)
            return jsonify({"message": "Menu data extracted and inserted successfully.", "records": len(filtered_menu)}), 200
        else:
            return jsonify({"message": "No valid menu items detected."}), 200

    except Exception as e:
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500

# if __name__ == "__main__":
#     if len(sys.argv) > 1 and sys.argv[1] == 'folder':
#         vender_id = 1
#         process_folder("menu1", mysql_config, vender_id)
#     else:
#         app.run(debug=False)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == 'folder':
        vender_id = 1
        process_folder("menu1", mysql_config, vender_id)
    else:
        port = int(os.environ.get("PORT", 5000))  # Render provides PORT env var
        app.run(host="0.0.0.0", port=port, debug=False)
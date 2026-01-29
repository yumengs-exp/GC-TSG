import json
import math
from geopy.distance import geodesic
import os
import json
from collections import Counter


def load_json(file_path):

    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def save_json(data, output_path):
    with open(output_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4, ensure_ascii=False)


def load_detections():

    with open("./Dataset/Detection(copenhagen).json", 'r', encoding='utf-8') as file:
        detections = [json.loads(line.strip()) for line in file if line.strip()]
    return detections

def load_signs():
 
    file_path = './Dataset/cleaned_signs_total.json'
    return load_json(file_path)


def load_cleaned_detections():
  
    with open("./Dataset/cleaned_detections.json", 'r', encoding='utf-8') as file:
        detections = [json.loads(line.strip()) for line in file if line.strip()]
    return detections

def load_cleaned_signs():
  
    file_path = './Dataset/cleaned_signs.json'
    signs= load_json(file_path)
    return list(signs.values())


def normalize_etl_cls(etl_cls):
   
    if ':' in etl_cls:
        base, text = etl_cls.split(':', 1)
    else:
        base, text = etl_cls, None
  
    base = base.replace('.', ',').lower()
    return base, text

def extract_digits(text):
  
    if not text:
        return None
    match = re.search(r'\d+', str(text))
    return match.group(0) if match else None

def normalize_heading(h):
    return (-h) % 360 if h is not None else None

def clean_detections_by_area_ratio(file_path='./Dataset/cleaned_detections_200_60.json',output_path='./Dataset/cleaned_detections_with_area_ratio.json',min_area_ratio=0.0001):
    detections = load_json(file_path)
    cleaned_detections = []

    for det in detections:
        etl_cls = det.get("etl_cls")
        if not etl_cls:
            continue

        sign_base, sign_text = normalize_etl_cls(etl_cls)
        heading = normalize_heading(det.get("heading"))
        width = det.get("width")
        height = det.get("height")
        img_width = det.get("img_width")
        img_height = det.get("img_height")
        det_lat = det.get("match_lat")
        det_lng = det.get("match_lng")
  
        if not all([width, height, img_width, img_height]):
            continue

        area_ratio = (width * height) / (img_width * img_height)
        if area_ratio < min_area_ratio:
            continue

        cleaned_detections.append(det)

    save_json(cleaned_detections, output_path)
def angle_difference(a, b):
    return min(abs(a - b), 360 - abs(a - b))
def calculate_bearing(lat1, lon1, lat2, lon2):
    dLon = math.radians(lon2 - lon1)
    lat1 = math.radians(lat1)
    lat2 = math.radians(lat2)
    y = math.sin(dLon) * math.cos(lat2)
    x = math.cos(lat1)*math.sin(lat2) - math.sin(lat1)*math.cos(lat2)*math.cos(dLon)
    return (math.degrees(math.atan2(y, x)) + 360) % 360

def clean_data():
    detections = load_detections()
    signs = load_signs()

  
    final_signs = []
    for sign in signs:
        sign_type = sign.get("sign_type", "").replace(',', ',').lower()
        sign_text_val = extract_digits(sign.get("sign_text"))
        sign_point = (sign.get("latitude"), sign.get("longitude"))

        if not all(sign_point):
            continue

        matched = False

        for det in detections:
            etl_cls = det.get("etl_cls")
            if not etl_cls:
                continue
            base, text = normalize_etl_cls(etl_cls)
            if base == 'e33,1' and base in sign_type:
                distance = geodesic(sign_point, det_point).meters
                if distance <= 30:
                    matched = True
                    break

            if base != sign_type:
                continue
            if text and sign_text_val and text != sign_text_val:
                continue

            det_point = (det.get("match_lat"), det.get("match_lng"))
            if not all(det_point):
                continue

            distance = geodesic(sign_point, det_point).meters
            if distance <= 30:
                matched = True
                break

        if matched:
            final_signs.append(sign)
    save_json(final_signs, './Dataset/cleaned_signs_30.json')


def load_cleaned_data():
   
    detection_file = './Dataset/cleaned_detections.json'
    sign_file = './Dataset/cleaned_signs.json'
    detections = load_json(detection_file)
    signs = load_json(sign_file)
    return detections, signs



def stat_detections(file_path='./Dataset/cleaned_detections_200_60.json'):
   
    detections = load_json(file_path)

    total = len(detections)
    type_counter = Counter()

    for det in detections:
        etl_cls = det.get('etl_cls', 'unknown').lower()
        type_counter[etl_cls] += 1

    print(f"total: {total} detection")
    print("etl_cls：")
    for etl_cls, count in type_counter.items():
        print(f"  {etl_cls}: {count}")

def stat_raw_detections(file_path='./Dataset/Detection(copenhagen).json'):

    detections = load_detections()
    total = len(detections)
    type_counter = Counter()

    for det in detections:
        etl_cls = det.get('etl_cls', 'unknown').lower()
        type_counter[etl_cls] += 1

    print(f"{total}  detection")
    print("etl_cls:")
    for etl_cls, count in type_counter.items():
        print(f"type:{etl_cls} | count: {count}")

def compare_removed_etl_cls(
    raw_path='./Dataset/Detection(copenhagen).json',
    cleaned_path='./Dataset/cleaned_detections_200_60.json'
):
    raw_detections = load_detections()
    cleaned_detections = load_json(cleaned_path)
    raw_total = len(raw_detections)
    clean_total = len(cleaned_detections)

    raw_counter = Counter(det.get('etl_cls', 'unknown').lower() for det in raw_detections)
    cleaned_types = set(det.get('etl_cls', 'unknown').lower() for det in cleaned_detections)
    print(f"raw_total: {raw_total} detection", f"clean_data:{clean_total}  detection")
    print("cleaned type:")
    print("-" * 40)
    removed = []
    for etl_cls, count in raw_counter.items():
        if etl_cls not in cleaned_types:
            print(f"type: {etl_cls}  | number: {count}")
            removed.append((etl_cls, count))

    if not removed:
        print("all type reserved")



def clean_sign_data(input_path='./Dataset/skilte_total.json', output_path='./Dataset/cleaned_signs_total.json'):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    cleaned_data = []

    for feature in data.get("features", []):
        geometry = feature.get("geometry")
        props = feature.get("properties", {})
        status = props.get("status")


        if geometry is None or status != "Opstillet":
            continue

        lon, lat = geometry.get("coordinates")
        uuid = props.get("uuid")

        if props.get("hovedtavle_1"):
            cleaned_data.append({
                "uuid": uuid,
                "longitude": lon,
                "latitude": lat,
                "sign_type": props.get("hovedtavle_1"),
                "sign_text": props.get("tekst_i_hovedtavle_1"),
                "category": "main_sign_1"
            })


        if props.get("hovedtavle_1_bagside"):
            cleaned_data.append({
                "uuid": uuid,
                "longitude": lon,
                "latitude": lat,
                "sign_type": props.get("hovedtavle_1_bagside"),
                "sign_text": None,
                "category": "main_sign_1_back"
            })


        for i in range(1, 4):
            undertavle = props.get(f"undertavle_1_{i}")
            if undertavle:
                cleaned_data.append({
                    "uuid": uuid,
                    "longitude": lon,
                    "latitude": lat,
                    "sign_type": undertavle,
                    "sign_text": props.get("tekst_i_undertavler_1"),
                    "category": f"sub_sign_1_{i}"
                })


        if props.get("hovedtavle_2"):
            cleaned_data.append({
                "uuid": uuid,
                "longitude": lon,
                "latitude": lat,
                "sign_type": props.get("hovedtavle_2"),
                "sign_text": props.get("tekst_i_hovedtavle_2"),
                "category": "main_sign_2"
            })

        for i in range(1, 4):
            undertavle = props.get(f"undertavle_2_{i}")
            if undertavle:
                cleaned_data.append({
                    "uuid": uuid,
                    "longitude": lon,
                    "latitude": lat,
                    "sign_type": undertavle,
                    "sign_text": props.get("tekst_i_undertavler_2"),
                    "category": f"sub_sign_2_{i}"
                })

  
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(cleaned_data, f, indent=2, ensure_ascii=False)

    print(f"✅ Cleaned data saved to: {output_path}")


def load_cleaned_data(path='./Dataset/cleaned_signs_total.json'):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)
    

import re

def check_main_sign_format(path='./Dataset/cleaned_signs_total.json'):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    pattern = re.compile(r'^[A-Z]+\d+(\.\d+)?$')
    results = {
        'valid': [],
        'invalid': []
    }

    for entry in data:
        sign = entry.get('main_sign_1')
        if sign is None:
            results['invalid'].append(sign)
        elif pattern.match(sign):
            results['valid'].append(sign)
        else:
            results['invalid'].append(sign)

    print(f"✅ Valid formats ({len(results['valid'])}): Sample -> {results['valid'][:5]}")
    print(f"❌ Invalid formats ({len(results['invalid'])}): Sample -> {results['invalid'][:5]}")

    return results


import statistics

def analyze_bbox_area_ratio(file_path='./Dataset/cleaned_detections.json'):
    with open(file_path, 'r', encoding='utf-8') as f:
        detections = json.load(f)

    area_ratios = []

    for det in detections:
        try:
            w = det['width']
            h = det['height']
            img_w = det['img_width']
            img_h = det['img_height']

            if w > 0 and h > 0 and img_w > 0 and img_h > 0:
                ratio = (w * h) / (img_w * img_h) * 100
                area_ratios.append(ratio)
        except (KeyError, ZeroDivisionError, TypeError):
            continue 

    if not area_ratios:
        print("no valid data")
        return

    max_ratio = max(area_ratios)
    min_ratio = min(area_ratios)
    median_ratio = statistics.median(area_ratios)
    mean_ratio = statistics.mean(area_ratios)

    print("bbx ratio:")
    print(f"max_ratio: {max_ratio:.4f}%")
    print(f"min_ratio: {min_ratio:.4f}%")
    print(f"median_ratio: {median_ratio:.4f}%")
    print(f"mean_ratio: {mean_ratio:.4f}%")

    return {
        "max": max_ratio,
        "min": min_ratio,
        "median": median_ratio,
        "mean": mean_ratio
    }


def normalize_signs(sign) -> str:
    sign_type = sign.get("sign_type", "").replace(',', '.').lower()
    sign_text_val = extract_digits(sign.get("sign_text"))
    if 'e33,1' in sign_type:
        return 'e33,1'
    if sign_text_val:
        return f'{sign_type}:{sign_text_val}'
    else:
        return sign_type


def top_etl_cls(file_path='./Dataset/cleaned_detections_200_60.json',
                sign_path='./Dataset/cleaned_signs_30.json',
                top_n=12):

    with open(file_path, 'r', encoding='utf-8') as f:
        detections = json.load(f)


    with open(sign_path, 'r', encoding='utf-8') as f:
        signs = json.load(f)
    sign_types = {
        normalize_signs(sign)
        for sign in signs
        if sign.get("sign_type")
    }


    etl_list = [
        etl for det in detections
        if (etl := det.get("etl_cls").lower()) in sign_types
    ]
    counter = Counter(etl_list)
    top_items = counter.most_common(top_n)

    return [item[0] for item in top_items]

    
if __name__ == "__main__":
    clean_data()
    # stat_detections()
    # stat_raw_detections()
    # compare_removed_etl_cls()
    # clean_sign_data()
    # check_main_sign_format()
    # analyze_bbox_area_ratio()
    
    # print(top_etl_cls())
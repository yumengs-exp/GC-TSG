import json
import math
from geopy.distance import geodesic
import os
import json
from collections import Counter
from geopandas import GeoDataFrame
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib.pyplot as plt


def load_json(file_path):

    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def save_json(data, output_path):
    with open(output_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4, ensure_ascii=False)


def load_detections():

    with open("./AAL_DATA/Detection(aal).json", 'r', encoding='utf-8') as file:
        detections = [json.loads(line.strip()) for line in file if line.strip()]
    return detections

def load_signs():

    file_path = './AAL_DATA/cleaned_signs_total.json'
    return load_json(file_path)


def load_cleaned_detections():

    with open("./AAL_DATA/cleaned_detections.json", 'r', encoding='utf-8') as file:
        detections = [json.loads(line.strip()) for line in file if line.strip()]
    return detections

def load_cleaned_signs():

    file_path = './AAL_DATA/cleaned_signs.json'
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

def clean_detections_by_area_ratio(file_path='./AAL_DATA/cleaned_detections_200_60.json',min_area_ratio=0.0004):
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

    save_json(cleaned_detections, './AAL_DATA/cleaned_detections_with_area_ratio.json')
def angle_difference(a, b):
    return min(abs(a - b), 360 - abs(a - b))
def calculate_bearing(lat1, lon1, lat2, lon2):
    dLon = math.radians(lon2 - lon1)
    lat1 = math.radians(lat1)
    lat2 = math.radians(lat2)
    y = math.sin(dLon) * math.cos(lat2)
    x = math.cos(lat1)*math.sin(lat2) - math.sin(lat1)*math.cos(lat2)*math.cos(dLon)
    return (math.degrees(math.atan2(y, x)) + 360) % 360

    
def process_sign(sign, detections):
    class_list = ['c62', 'c55:50', 'n42.3', 'e56', 'e55', 'c61', 'b11', 'e33.1', 'c55:60', 'd15.3', 'd11.3', 'c56:60']
    sign_type = sign.get("sign_type", "").lower()
    sign_text = extract_digits(sign.get("sign_text"))

    if sign_text:
        sign["sign_type"] = f"{sign_type}:{sign_text}"
        sign["sign_text"] = ""
        sign_type = sign["sign_type"]
    if sign_type not in class_list:
        return None
    
    sign_point = (sign.get("latitude"), sign.get("longitude"))
    if not all(sign_point):
        return None
    
    return sign
   

def process_detection(det, final_signs):
    class_list = ['c62', 'c55:50', 'n42.3', 'e56', 'e55', 'c61', 'b11', 'e33.1', 'c55:60', 'd15.3', 'd11.3', 'c56:60']
    etl_cls = det.get("etl_cls")
    if not etl_cls or etl_cls not in class_list:
        return None

    heading = normalize_heading(det.get("heading"))
    det_lat = det.get("match_lat")
    det_lng = det.get("match_lng")

    if det_lat is None or det_lng is None or heading is None:
        return None
    
    return det

   

def clean_data():
    detections = load_detections()
    signs = load_signs()

   
    final_signs = []
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_sign, sign, detections) for sign in signs]
        for future in as_completed(futures):
            result = future.result()
            if result:
                final_signs.append(result)

    save_json(final_signs, './AAL_DATA/cleaned_signs_skip.json')

  
    cleaned_detections = []
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_detection, det, final_signs) for det in detections]
        for future in as_completed(futures):
            result = future.result()
            if result:
                cleaned_detections.append(result)

    save_json(cleaned_detections, './AAL_DATA/cleaned_detections_skip.json')

def load_cleaned_data():

    detection_file = './AAL_DATA/cleaned_detections.json'
    sign_file = './AAL_DATA/cleaned_signs.json'
    detections = load_json(detection_file)
    signs = load_json(sign_file)
    return detections, signs



def stat_detections(file_path='./AAL_DATA/cleaned_detections_200_60.json'):
  
    detections = load_json(file_path)

    total = len(detections)
    type_counter = Counter()

    for det in detections:
        etl_cls = det.get('etl_cls', 'unknown').lower()
        type_counter[etl_cls] += 1

    print(f"total: {total}  detection")
    print(" etl_cls:")
    for etl_cls, count in type_counter.items():
        print(f"  {etl_cls}: {count}")

def stat_raw_detections(file_path='./AAL_DATA/Detection(aal).json'):

    detections = load_detections()
    total = len(detections)
    type_counter = Counter()

    for det in detections:
        etl_cls = det.get('etl_cls', 'unknown').lower()
        type_counter[etl_cls] += 1

    print(f"total {total} detection")
    print("etl_cls:")
    for etl_cls, count in type_counter.items():
        print(f"type:{etl_cls} | count: {count}")

def compare_removed_etl_cls(
    raw_path='./AAL_DATA/Detection(aal).json',
    cleaned_path='./AAL_DATA/cleaned_detections_200_60.json'
):
    raw_detections = load_detections()
    cleaned_detections = load_json(cleaned_path)
    raw_total = len(raw_detections)
    clean_total = len(cleaned_detections)
 
    raw_counter = Counter(det.get('etl_cls', 'unknown').lower() for det in raw_detections)
    cleaned_types = set(det.get('etl_cls', 'unknown').lower() for det in cleaned_detections)
    print(f"raw_total: {raw_total} detection", f"clean_total: {clean_total} detection")
    print("after cleaning, removed etl_cls:")
    print("-" * 40)
    removed = []
    for etl_cls, count in raw_counter.items():
        if etl_cls not in cleaned_types:
            print(f"type: {etl_cls}  | count: {count}")
            removed.append((etl_cls, count))

    if not removed:
        print("all types are preserved after cleaning.")


def clean_sign_data(input_path='./AAL_DATA/skilte_total.json', output_path='./AAL_DATA/cleaned_signs_total.json'):
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


def load_cleaned_data(path='./AAL_DATA/cleaned_signs_total.json'):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)
    

import re

def check_main_sign_format(path='./AAL_DATA/cleaned_signs_total.json'):
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

def analyze_bbox_area_ratio(file_path='./AAL_DATA/cleaned_detections.json'):
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
        print("no valid area ratios found.")
        return

    max_ratio = max(area_ratios)
    min_ratio = min(area_ratios)
    median_ratio = statistics.median(area_ratios)
    mean_ratio = statistics.mean(area_ratios)

    print("bbox ratio:")
    print(f"max_ratio: {max_ratio:.4f}%")
    print(f"min_ratio: {min_ratio:.4f}%")
    print(f"median: {median_ratio:.4f}%")
    print(f"mean: {mean_ratio:.4f}%")

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


def top_etl_cls(file_path='./AAL_DATA/cleaned_detections_skip.json',
                sign_path='./AAL_DATA/cleaned_signs_skip.json',
                top_n=12):

    with open(file_path, 'r', encoding='utf-8') as f:
        detections = json.load(f)


    with open(sign_path, 'r', encoding='utf-8') as f:
        signs = json.load(f)
    sign_types = [
        sign.get("sign_type")
        for sign in signs
    ]

    sign_types = set(sign_types)

    detection_types = [
        detection.get("etl_cls")
        for detection in detections
    ]

    detection_types = set(detection_types)

    etl_list = [
        types
        for types in sign_types
        if types in detection_types
    ]

    return sign_types
    

import os
import matplotlib.pyplot as plt
import contextily as ctx
import geopandas as gpd
from shapely.geometry import Point



def visualize_by_class_modes(detection_path='./AAL_DATA/Detection(aal).json',
                sign_path='./AAL_DATA/cleaned_signs_total.json'):

    detections = load_detections()

    with open(sign_path, 'r', encoding='utf-8') as f:
        signs = json.load(f)

   
    output_dir = "output/AAL/TD"
    os.makedirs(output_dir, exist_ok=True)


    detection_classes = set(d['etl_cls'] for d in detections)
    sign_classes = set(s['sign_type'] for s in signs)
    common_classes = detection_classes & sign_classes

    for cls in common_classes:
    
        det_points = [
            Point(d['match_lng'], d['match_lat'])
            for d in detections if d['etl_cls'] == cls
        ]
        det_gdf = gpd.GeoDataFrame(geometry=det_points, crs="EPSG:4326")

      
        sign_points = [
            Point(s['longitude'], s['latitude'])
            for s in signs if s['sign_type'] == cls
        ]
        sign_gdf = gpd.GeoDataFrame(geometry=sign_points, crs="EPSG:4326")

        det_gdf = det_gdf.to_crs(epsg=3857)
        sign_gdf = sign_gdf.to_crs(epsg=3857)

        fig, ax = plt.subplots(figsize=(10, 10))

        if not det_gdf.empty:
            det_gdf.plot(ax=ax, color='blue', marker='o', label='Detections', alpha=0.6)

        if not sign_gdf.empty:
            sign_gdf.plot(ax=ax, color='red', marker='^', label='Signs', alpha=0.8)

  
        try:
            ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)
        except Exception as e:
            print(f"{e}")

        ax.legend()
        ax.set_title(f"Class: {cls}")
        ax.set_axis_off()

   
        filepath = os.path.join(output_dir, f"{cls}.png")
        plt.savefig(filepath, bbox_inches='tight', dpi=300)
        plt.close()

    print(f"{output_dir}")



    
if __name__ == "__main__":
    clean_data()
    # stat_detections()
    # stat_raw_detections()
    # compare_removed_etl_cls()
    # clean_sign_data()
    # check_main_sign_format()
    # analyze_bbox_area_ratio()
    
    print(top_etl_cls())
    # visualize_by_class_modes()
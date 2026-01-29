import json
import uuid
import os


input_files = {
    'train': './AAL_DATA/training_truth.geojson',
    'val': './AAL_DATA/validation_truth.geojson',
    'test': './AAL_DATA/testing_truth.geojson'
}


output_data_path = './AAL_DATA/cleaned_signs_total.json'
output_split_index_path = './AAL_DATA/signs_split_index.json'


cleaned_list = []
split_index = {}

for split_name, file_path in input_files.items():
    with open(file_path, 'r', encoding='utf-8') as f:
        geojson_data = json.load(f)

    for feature in geojson_data['features']:
        prop = feature['properties']
        geom = feature['geometry']
        coords = geom['coordinates']
        sign_type = prop.get('label_name', '').lower()
        sign_text = prop.get('sign_text')

   
        if sign_text and sign_type not in ['e55', 'e56']:
            sign_type = f"{sign_type}:{sign_text}"
            sign_text = None
        elif sign_text and sign_type in ['e55', 'e56']:
            sign_text = None

        longitude, latitude = coords[0], coords[1]
        sign_uuid = str(uuid.uuid4())
        category = "main_sign_1"

        cleaned_entry = {
            "uuid": sign_uuid,
            "longitude": longitude,
            "latitude": latitude,
            "sign_type": sign_type if sign_type else None,
            "sign_text": sign_text,
            "category": category
        }

        cleaned_list.append(cleaned_entry)
        split_index[sign_uuid] = split_name


with open(output_data_path, 'w', encoding='utf-8') as f:
    json.dump(cleaned_list, f, ensure_ascii=False, indent=2)


with open(output_split_index_path, 'w', encoding='utf-8') as f:
    json.dump(split_index, f, ensure_ascii=False, indent=2)

print(f"{len(cleaned_list)}")
print(f" {output_data_path}")
print(f" {output_split_index_path}")


import json
from datetime import datetime
from collections import defaultdict


input_files = {
    'train': './AAL_DATA/training.geojson',
    'val': './AAL_DATA/validation.geojson',
    'test': './AAL_DATA/testing.geojson'
}

output_path = './AAL_DATA/Detection(aal).json'
index_output_path = './AAL_DATA/detection_split_index.json'


all_features = []
split_index = {}  

for split_name, file_path in input_files.items():
    with open(file_path, 'r', encoding='utf-8') as f:
        geojson_data = json.load(f)
        for feature in geojson_data['features']:
            feature['split'] = split_name 
            all_features.append(feature)


trip_groups = defaultdict(list)
for feature in all_features:
    prop = feature['properties']
    capture_time = prop.get("image_capture_time")
    if capture_time is None:
        continue
    trip_no = prop.get("trip_no")
    trip_groups[trip_no].append((capture_time, feature))

converted = []
record_counter = 0

for trip_no, feature_list in trip_groups.items():
    feature_list.sort(key=lambda x: x[0])  

    for img_seq_id, (capture_time, feature) in enumerate(feature_list, start=1):
        prop = feature['properties']
        coords = feature['geometry']['coordinates']
        try:
            dt = datetime.strptime(capture_time, "%Y-%m-%dT%H:%M:%S")
            date_no = int(dt.strftime("%Y%m%d"))
            time_no = int(dt.strftime("%H%M%S"))
        except:
            date_no = None
            time_no = None

        record = {
            "trip_no": prop.get("trip_no"),
            "img_seq_id": img_seq_id,
            "detection_no": prop.get("object_no"),
            "date_no": date_no,
            "time_no": time_no,
            "device_cls": prop.get("label_name"),
            "etl_cls": prop.get("label_name"),
            "device_score": 1.0,
            "etl_score": 1.0,
            "x": 100,
            "y": 200,
            "width": prop.get("width"),
            "height": prop.get("height"),
            "img_width": 3840,
            "img_height": 2160,
            "speed": prop.get("speed"),
            "heading": prop.get("obj_heading"),
            "alt": 42.0,
            "alt_accuracy": 1.0,
            "gps_accuracy": 1.0,
            "raw_lng": coords[0],
            "raw_lat": coords[1],
            "match_lng": coords[0],
            "match_lat": coords[1]
        }

      
        record_id = f"rec_{record_counter}"
        split_index[record_id] = feature['split']
        record["record_id"] = record_id
        record_counter += 1

        converted.append(record)

with open(output_path, 'w', encoding='utf-8') as f:
    for item in converted:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')


with open(index_output_path, 'w', encoding='utf-8') as f:
    json.dump(split_index, f, ensure_ascii=False, indent=2)

print(f" {len(converted)}  {output_path}")
print(f"{index_output_path}")

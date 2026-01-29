import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Set

from pandas import DataFrame
from shapely.geometry import Point
from shapely.strtree import STRtree
from shapely.geometry.base import BaseGeometry
import Predict
from Utils.structure import Position

DEFAULT_CLASSES = {'c61', 'd15.3', 'e17', 'c11.2', 'n42.2', 'c22.1', 'c19', 'n43', 'c11.1', 'b11', 'c55:60', 'c62'}
DATA_PATH = Path(__file__).parents[1].joinpath('data')
AALBORG_DATA_PATH = DATA_PATH.joinpath('aalborg')
HP_TUNING_DATA_PATH = DATA_PATH.joinpath('hp_tuning')

logger = logging.getLogger(Path(__file__).stem)


def map_cleaned_detection(position: Position):
  
    heading_deg = position.properties.get('heading')
    return {
        'id': position.properties.get('detection_no'),
        'geom': Point(position.point),
        'trip_id': position.properties.get('trip_no'),
        'img_seq_id': position.properties.get('img_seq_id'),
        'detection_no': position.properties.get('detection_no'),
        'date_no': position.properties.get('date_no'),
        'time_no': position.properties.get('time_no'),
        'classifier': position.properties.get('etl_cls'),
        'device_cls': position.properties.get('device_cls'),
        'speed': position.properties.get('speed'),
        'heading': math.radians((-heading_deg) % 360) if heading_deg is not None else 1e-2,
        'score': position.properties.get('etl_score'),
        'device_score': position.properties.get('device_score'),
        'width': position.properties.get('width'),
        'height': position.properties.get('height'),
        'img_width': position.properties.get('img_width'),
        'img_height': position.properties.get('img_height'),
        'x': position.properties.get('x'),
        'y': position.properties.get('y'),
        'alt': position.properties.get('alt'),
        'gps_accuracy': position.properties.get('gps_accuracy'),
        'match_lat': position.properties.get('match_lat'),
        'match_lng': position.properties.get('match_lng'),
        'raw_lat': position.properties.get('raw_lat'),
        'raw_lng': position.properties.get('raw_lng')
    }

def load_gomap_detections(filepath: str, classes: Set[str] = DEFAULT_CLASSES) -> DataFrame:
 
    detections = Predict.load_json_detections(filepath)  
    df = DataFrame(map(map_cleaned_detection, detections))

    logger.info('Loaded %d detections', len(df))


    if classes:
        df = df[df['classifier'].str.lower().isin({c.lower() for c in classes})]
        logger.info('%d detections remain after applying class filter', len(df))
        logger.debug(df.groupby('classifier')['classifier'].agg('count'))

    return df


def load_aal_detections(classes: Set[str] = DEFAULT_CLASSES, min_score=0.8, max_distance=30) -> DataFrame :
    return load_gomap_detections(AALBORG_DATA_PATH.joinpath('traffic_sign_detections.geojson'), classes, min_score, max_distance)


def load_gomap_train_detections(classes: Set[str] = DEFAULT_CLASSES, min_score=0.8, max_distance=30) -> DataFrame :
    return load_gomap_detections(HP_TUNING_DATA_PATH.joinpath('training.geojson'), classes, min_score, max_distance)


def load_gomap_validation_detections(classes: Set[str] = DEFAULT_CLASSES, min_score=0.8, max_distance=30) -> DataFrame :
    return load_gomap_detections(HP_TUNING_DATA_PATH.joinpath('validation.geojson'), classes, min_score, max_distance)


def load_gomap_test_detections(file_path='./Dataset/cleaned_detections_with_area_ratio.json', classes: Set[str] = DEFAULT_CLASSES) -> DataFrame :
    return load_gomap_detections(file_path, classes)


def load_cop_detections(classes: Set[str] = DEFAULT_CLASSES) -> DataFrame :
    return load_gomap_detections('./Dataset/cleaned_detections_200_60.json', classes)

def load_aal_detections(classes: Set[str] = DEFAULT_CLASSES) -> DataFrame :
    return load_gomap_detections('./AAL_DATA/cleaned_detections_200_60.json', classes)

def load_detections(file_name = './Dataset/cleaned_detections_200_60.json',classes: Set[str] = DEFAULT_CLASSES, ) -> DataFrame :
    return load_gomap_detections(file_name, classes)

def extract_digits(text):
  
    import re
    if not text:
        return None
    match = re.search(r'\d+', str(text))
    return match.group(0) if match else None

def _map_cleaned_truth_heading(position: Position):
 
    if 'map_heading' in position.properties:
        position.properties['heading'] = math.radians(position.properties['map_heading'])
    else:
        position.properties['heading'] = 1e-2
    return position

def load_gomap_truths(filepath: str, classes: Set[str] = DEFAULT_CLASSES) -> STRtree:

    def normalize_label_name(position: Position) -> Position:
        label = position.properties.get('sign_type', '').replace(',', '.').lower()
        if label in {'c55', 'c56'}:
            digits = extract_digits(position.properties.get('sign_text'))
            if digits:
                label = f"{label}:{digits}"
        if 'c33.1' in label:
            label = 'c33.1'

        position.properties['label_name'] = label
        return position

    truths = Predict.load_json_signs(filepath)  
    truths = list(truths)
    logger.info('Loaded %d truths', len(truths))

    truths = map(normalize_label_name, truths)
    truths = [x for x in truths if x.properties.get('label_name') in {c.lower() for c in classes}]
    logger.info('%d truths remain after applying class filter', len(truths))

    truths = list(map(_map_cleaned_truth_heading, truths))
    
    for i, truth in enumerate(truths):
        pt = getattr(truth, "point", None)
        if not isinstance(pt, BaseGeometry):
            raise TypeError(f"Item {i} has invalid 'point': {pt} (type: {type(pt)})")

    truths_points = [truth.point for truth in truths] 
    point_to_truth = {id(truth.point): truth for truth in truths}
    return STRtree(truths_points), point_to_truth, truths_points

def load_aal_truths(classes: Set[str] = DEFAULT_CLASSES) -> STRtree:
    return load_gomap_truths(AALBORG_DATA_PATH.joinpath('traffic_sign_ground_truth.geojson'), classes)


def load_gomap_test_truths(classes: Set[str] = DEFAULT_CLASSES) -> STRtree:
    return load_gomap_truths('./Dataset/cleaned_signs_30.json', classes)


def load_test_truths(groundtruth_file_name='./Dataset/cleaned_signs_30.json',classes: Set[str] = DEFAULT_CLASSES) -> STRtree:
    return load_gomap_truths(groundtruth_file_name, classes)


import json
import math
from typing import Callable, Iterable, List

# import geojson
# import pyproj
# from geojson import Feature, FeatureCollection, Point
from pandas import DataFrame, Series
from shapely.ops import unary_union, transform

# from GoMapClustering.angle_balancing_dbscan import AngleBalancingDBSCAN
# from GoMapClustering.dbscan_fcm4dd import DBSCANFCM4DD
# from GoMapClustering.angle_metric_dbscan import AngleMetricDBSCAN
from GoMapClustering.dbscan_x2 import DBSCANx2
from GoMapClustering.dbscan import DBSCAN
import pandas as pd

# from GoMapClustering import DBSCAN, DBSCANFCM4DD, DBSCANx2, AngleBalancingDBSCAN, AngleMetricDBSCAN
from GoMapClustering.base import GoMapClusterMixin
import pandas as pd
import numpy as np
import math
import networkx as nx
from typing import List, Callable, Dict

PI = math.pi
TWO_PI = 2 * math.pi
from shapely.geometry import Point
from typing import Any, Dict
import numpy as np
from numpy import ndarray
from pandas import DataFrame
from Utils.structure import Position
from sklearn.base import ClusterMixin
from pyproj import Transformer
from old_dataclean.detection2traj import track_detections_in_group


transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

def load_json_detections(filepath: str) -> Iterable[Position]:

    with open(filepath, 'r', encoding='utf-8') as f:
        detections = json.load(f)

    for det in detections:
        lng = det.get('match_lng')
        lat = det.get('match_lat')

        if lng is None or lat is None:
            continue


        x, y = transformer.transform(lng, lat)
        pos = Position(Point(x, y))  
        pos.properties = det  
        yield pos

class AngleIndexer:
    """Group angeles into slices in the unit circle.
    """

    def __init__(self, size: int = 36) -> None:
        self.slice = TWO_PI / size

    def index(self, angle: float) -> int:
        return math.floor(((angle + self.slice / 2) / self.slice) % (TWO_PI / self.slice))


def compute_cluster_centroid(cluster: List[Point]) -> Point:
    return unary_union(cluster).centroid


def signed_angle_diff(source: float, target: float) -> float:
  
    return ((target - source + PI) % TWO_PI) - PI


# def load_geoJson(filepath: str, sourceCrs: str, targetCrs: str) -> Iterable[Position]:
#     source = pyproj.CRS(sourceCrs)
#     target = pyproj.CRS(targetCrs)
#     project = pyproj.Transformer.from_crs(source, target, always_xy=True)

#     with open(filepath, 'r') as file:
#         features = json.load(file)['features']

#         for feature in features:
#             if feature['geometry'] is None:
#                 continue

#             geom = Position(feature['geometry']['coordinates'])
#             geom = transform(project.transform, geom)
#             geom.properties = feature['properties']

#             yield geom



def load_json_signs(filepath: str) -> Iterable[Position]:
  
    with open(filepath, 'r', encoding='utf-8') as f:
        signs = json.load(f)

    for sign in signs:
        lng = sign.get('longitude')
        lat = sign.get('latitude')

        if lng is None or lat is None:
            continue

        # 经纬度 → 米
        x, y = transformer.transform(lng, lat)
        pos = Position(Point(x, y))
        pos.properties = sign
        yield pos

# def dump_geoJson(filepath: str, sourceCrs: str, targetCrs: str, points: Iterable[Position]):
#     source = pyproj.CRS(sourceCrs)
#     target = pyproj.CRS(targetCrs)
#     project = pyproj.Transformer.from_crs(source, target, always_xy=True)

#     features = []
#     for point in points:
#         transformed_point = transform(project.transform, point)
#         feature_point = Point(
#             [transformed_point.x, transformed_point.y], precision=15)
#         feature = Feature(None, feature_point, point.properties)
#         features.append(feature)

#     featureCollection = FeatureCollection(features)

#     with open(filepath, 'w') as file:
#         file.write(geojson.dumps(featureCollection))


def cluster(
    df: DataFrame,
    cluster_algo: GoMapClusterMixin
) -> DataFrame:
    rv = DataFrame()
    for _, group in df.groupby('classifier'):
        group['cid'] = cluster_algo.cluster(group)
        rv = pd.concat([rv, group], ignore_index=True)

    return rv


def aggregate(df: DataFrame, cluster_singularity: Callable[[List[Position]], Position]) -> DataFrame:
    size = 36
    indexer = AngleIndexer(size)
    df['angle_index'] = [indexer.index(x) for x in df['heading']]
    df['index_heading'] = [x * TWO_PI / size for x in df['angle_index']]

    return df\
        .groupby(['cid', 'classifier'], as_index=False)\
        .agg(
            geom=('geom', cluster_singularity),
            avg_score=('score', 'mean'),
            avg_speed=('speed', 'mean'),
            avg_heading=('heading', 'mean'),
            heading=('index_heading', lambda x: Series.mode(x)[0]),
            count=('geom', 'size'),
            trip_count=('trip_id', 'nunique')
        ).set_index(['cid', 'classifier'])

def location_predict(df: DataFrame, cluster_singularity: Callable[[List[Position]], Position]) -> DataFrame:
    size = 36
    indexer = AngleIndexer(size)
    df['angle_index'] = [indexer.index(x) for x in df['heading']]
    df['index_heading'] = [x * TWO_PI / size for x in df['angle_index']]

    return df\
        .groupby(['cid', 'classifier'], as_index=False)\
        .agg(
            geom=('geom', cluster_singularity),
            avg_score=('score', 'mean'),
            avg_speed=('speed', 'mean'),
            avg_heading=('heading', 'mean'),
            heading=('index_heading', lambda x: Series.mode(x)[0]),
            count=('geom', 'size'),
            trip_count=('trip_id', 'nunique')
        ).set_index(['cid', 'classifier'])


def get_predictions(
    df: DataFrame,
    cluster_algo: GoMapClusterMixin,
    cluster_singularity: Callable[[List[Position]], Position]
) -> DataFrame:
    # Cluster the data
    df = cluster(df, cluster_algo)
    # Remove outliers
    df = df[df['cid'] > -1]
    # Aggregate the data
    return aggregate(df, cluster_singularity)



def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    r = 6371000
    return c * r


def compute_detection_similarity(det1, det2, loc_thresh=15, angle_thresh=15):

    if det1['classifier'].lower() != det2['classifier'].lower():
        return 0


    loc_dist = haversine(det1['match_lng'], det1['match_lat'],
                         det2['match_lng'], det2['match_lat'])
    angle_diff = abs(det1.get('heading', 0) - det2.get('heading', 0))

    if loc_dist <= loc_thresh and angle_diff <= angle_thresh:
        return 1
    else:
        return 0



def compute_trajectory_similarity(traj1, traj2, trip_sim_matrix, weight_trip=0.5, weight_det=0.5):
    matched_pairs = []

    for d1 in traj1:
        for d2 in traj2:
            if compute_detection_similarity(d1, d2):  # 确认匹配位置角度
                a1 = (d1['width'] * d1['height']) / (d1['img_width'] * d1['img_height'])
                a2 = (d2['width'] * d2['height']) / (d2['img_width'] * d2['img_height'])
                matched_pairs.append((a1, a2))


    if not matched_pairs:
        sim_det = 0.0
    else:
        diffs = [abs(a1 - a2) / max(a1, a2) if max(a1, a2) > 0 else 0.0 for a1, a2 in matched_pairs]
        sim_det = 1.0 - sum(diffs) / len(diffs)


    trip_set = list(set([d['trip_id'] for d in traj1 + traj2]))
    sim_trip = 0.0
    if len(trip_set) == 2:
        t1, t2 = trip_set
        try:
            sim_trip = trip_sim_matrix.at[t1, t2]
        except KeyError:
            sim_trip = 0.0

    return weight_det * sim_det + weight_trip * sim_trip
    # return sim_trip * sim_det


def improved_predict(
    df: pd.DataFrame,
    cluster_algo,
    cluster_singularity: Callable[[List[dict]], dict],
    params: dict = {
        'w_size': 1.0,           
        'w_gps': 1.0,            
        'gps_threshold': 150.0,   
        'time_threshold': 30,    
        'beta': 8.0,             
        'expected_area_factor': 0.5, 
        'gps_norm_last': 10.0,   
        'cost_threshold': 5    
    },
    sim_threshold: float = 0.6
) -> pd.DataFrame:
    
    trip_sim_matrix = pd.read_pickle('./Dataset/trip_similarity_matrix.pkl')


    all_clustered = []
    for _, group in df.groupby('classifier'):
        group = group.copy()
        group['cid'] = cluster_algo.cluster(group)

        max_cid = group['cid'].max()
        outliers = group[group['cid'] == -1]
        group = group[group['cid'] != -1]
        for i, idx in enumerate(outliers.index, start=1):
            group.loc[idx] = outliers.loc[idx]
            group.at[idx, 'cid'] = max_cid + i
        all_clustered.append(group)

    clustered_df = pd.concat(all_clustered, ignore_index=True)
    new_clustered_rows = []

    for classifier, group in clustered_df.groupby('classifier'):
        final_detections = []
        cid_counter = 0  # 全局 cid 编号器

        for classifier, group in clustered_df.groupby('classifier'):
            for original_cid, cid_group in group.groupby('cid'):
                cid_group = cid_group.sort_values('img_seq_id')
                tracks = track_detections_in_group(cid_group, params)
                trajectories = list(tracks.values())

                G = nx.Graph()
                for i, t1 in enumerate(trajectories):
                    G.add_node(i)
                    for j in range(i + 1, len(trajectories)):
                        t2 = trajectories[j]
                        sim = compute_trajectory_similarity(t1, t2, trip_sim_matrix)
                        if sim >= sim_threshold:
                            G.add_edge(i, j)

                for component in nx.connected_components(G):
                    for idx in component:
                        for det in trajectories[idx]:
                            det_copy = det.copy()
                            det_copy['cid'] = cid_counter  # 使用全局编号
                            det_copy['original_cid'] = original_cid  # 可选：保留旧cid
                            det_copy['classifier'] = classifier
                            final_detections.append(det_copy)
                    cid_counter += 1  # 下一个聚簇编号递增
        new_clustered_rows.extend(final_detections)
    df = pd.DataFrame(new_clustered_rows)
    return aggregate(df, cluster_singularity)

    


def get_approaches() -> List[GoMapClusterMixin]:
    return [
        # AngleMetricDBSCAN(10.7, math.radians(27.5), 2),
        # AngleBalancingDBSCAN(11, math.radians(40.8), 2),
        # DBSCANx2(13.5, math.radians(34.2), 2),
        # DBSCANFCM4DD(
        #     max_spatial_distance=10.5,
        #     min_samples=2,
        #     c=2,
        #     m=8.3,
        #     max_iterations=180,
        #     min_improvement=0.48199954255368505,
        #     seed=1337
        # ),
        # #! COP
        # DBSCAN(10, 1)
        #! AAL
        DBSCAN(3.9, 2)
    ]
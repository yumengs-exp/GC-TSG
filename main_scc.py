import dataloader
from Evaluator import GoMapEvaluator
# from Split.GNN import run_graphcl_clustering, run_dbscan_clustering
from geopandas import GeoDataFrame
from Split.scc import run_two_clustering
import os
import matplotlib.pyplot as plt
from geopandas import GeoDataFrame
import contextily as ctx
from Geolocating.jiaodian import predict_cluster_geometries
from collections import Counter
import pandas as pd
from Geolocating.Residual import train_predict
from Geolocating.Residual import train_predict
from output import visualize_by_class_modes, print_evaluation_metrics, print_evaluation_metrics_by_class, visualize_diffs_by_class_skipfilter,get_evaluation_metrics





from typing import Callable, List
from pandas import DataFrame, Series
import pandas as pd
import Predict

TWO_PI = 2 * 3.141592653589793


class AngleIndexer:
    def __init__(self, size: int):
        self.size = size

    def index(self, heading_rad: float) -> int:
        return int((heading_rad % TWO_PI) / TWO_PI * self.size)


def aggregate(df: DataFrame, cluster_singularity: Callable[[List], object]) -> DataFrame:
    size = 36
    indexer = AngleIndexer(size)
    df['angle_index'] = [indexer.index(x) for x in df['heading']]
    df['index_heading'] = [x * TWO_PI / size for x in df['angle_index']]

    return df \
        .groupby(['cid', 'classifier'], as_index=False) \
        .agg(
        geom=('geom', cluster_singularity),
        avg_score=('score', 'mean'),
        avg_speed=('speed', 'mean'),
        avg_heading=('heading', 'mean'),
        heading=('index_heading', lambda x: Series.mode(x)[0]),
        count=('geom', 'size'),
        trip_count=('trip_id', 'nunique')
    ).set_index(['cid', 'classifier'])


if __name__ == "__main__":
    # Clean the data

    # ! COP DATASET
    classes_list = {'c61',  'b11', 'c11.2'}
    detection_file_name = './Dataset/cleaned_detections_200_60.json'
    groundtruth_file_name = './Dataset/cleaned_signs_30.json'
    dataset = 'COP'
    best_para = [4, 13]

    # ! AAL DATASET
    # classes_list = { 'b11', 'd15.3', 'c61', 'c55:60',  'c55:50'}
    # detection_file_name = './AAL_DATA/cleaned_detections_skip.json'
    # groundtruth_file_name = './AAL_DATA/cleaned_signs_skip.json'
    # dataset = 'AAL'
    # best_para = [1, 11]

    detections = dataloader.load_detections(detection_file_name, classes_list)
    truths, point_to_truth, truth_points = dataloader.load_test_truths(groundtruth_file_name, classes_list)

    grouped = detections.groupby('classifier')

    results = []
    detincluster_counter = []
    cluster_counter = 0
    result_dic = {}
    for classifier, group in grouped:
        
        # aal
        # clustered = run_two_clustering(
        #         group,
        #         hidden_dim=64,
        #         dataset_name=dataset,
        #
        #         init_min_cluster_size=5,
        #       
        #         size_factor=3.0,            
        #         diam_thresh=130.0,          
        #         heading_disp_thresh=0.8,   
        #         sub_min_cluster_size=1,     
        #    
        #         neighbor_radius=80.0,       
        #         heading_opposite_deg=90.0, 
        #         heading_similar_deg=85.0,  
        #         emb_merge_thresh=0.4,
        #         proj_sep_thresh=0.8,
        #         area_consistency_min=0.1,
        #         max_split_merge_rounds=10
        #     )

        # cph
        clustered = run_two_clustering(
                        group,
                        hidden_dim=64,
                        dataset_name=dataset,
             
                        init_min_cluster_size=2,
                  
                        size_factor=0,          
                        diam_thresh=3,        
                        heading_disp_thresh=0.1,   
                        sub_min_cluster_size=1,    
                      
                        neighbor_radius=0.5,       
                        heading_opposite_deg=90.0, 
                        heading_similar_deg=70.0,  
                        emb_merge_thresh=0.8,
                        proj_sep_thresh=0.8,
                        area_consistency_min=0.1,
                   
                        max_split_merge_rounds=10
                    )
       
        clustered = clustered.copy()
        
        # # ! for cph
        # max_cid = clustered.loc[clustered['cluster_id'] != -1, 'cluster_id'].max()
        # new_cid = max_cid + 1  

        #
        # new_cids = []
        # for idx, row in clustered.iterrows():
        #     if row['cluster_id'] == -1:
        #         new_cids.append(new_cid)
        #         new_cid += 1
        #     else:
        #         new_cids.append(row['cluster_id']) 

        # clustered['cid'] = new_cids
        # results.append(clustered)
        # cluster_counts = clustered['cid'].value_counts()
        # detincluster_counter.extend(cluster_counts.tolist())  

     
        #############################1
        #!for aal
        clustered = clustered.copy()
        clustered['cid'] = clustered['cluster_id']

        result_dic[classifier] = clustered




    if dataset == 'COP':
        gdf_train, left_truth_points, left_point_to_truth, left_search_tree, predictions, right_truth_points, right_point_to_truth, right_search_tree,gdf_test1, right_truth_points1, right_point_to_truth1, right_search_tree1  = train_predict(result_dic, truth_points, point_to_truth,truths,max_offset=best_para[0],random_seed=best_para[1],penalty_factor=0.1)
        
       
    if dataset == 'AAL':
        gdf_train, left_truth_points, left_point_to_truth, left_search_tree, predictions, right_truth_points, right_point_to_truth, right_search_tree,gdf_test1, right_truth_points1, right_point_to_truth1, right_search_tree1  = train_predict(result_dic, truth_points, point_to_truth,truths,max_offset=best_para[0],random_seed=best_para[1],penalty_factor=0.3)
            
    evaluator = GoMapEvaluator(predictions, right_search_tree, right_point_to_truth, right_truth_points,dataset_name= dataset)

    print_evaluation_metrics(evaluator)

import math
from typing import List, Tuple, Union

import numpy as np
from geopandas import GeoDataFrame
from numpy import ndarray
from pandas import DataFrame
from pandas.core.series import Series
from shapely.geometry import Point
from shapely.strtree import STRtree

import Predict
from Utils.structure import Position
from pyproj import Geod
from scipy.optimize import linear_sum_assignment
import pandas as pd
from typing import Union, Tuple

class GoMapEvaluator:
    def __init__(self, predictions: DataFrame, truths: STRtree, point_to_truth, truth_points,dataset_name='AAL') -> None:
        self.truths = truths
        predictions = self.merge_predictions(predictions)
        self.predictions = predictions
        self.point_map = point_to_truth
        self.truth_geoms = truth_points
    
        predictions = GeoDataFrame(
            predictions[['geom', 'heading']],
            crs='epsg:3044',
            geometry='geom'
        )
        self.visited = set()  
        if dataset_name=='COP':
            self.data: GeoDataFrame = self.__get_truths_c(predictions)\
                .set_index(predictions.index)\
                .join(predictions)\
                .dropna()
        else :
            self.data: GeoDataFrame = self.__get_truths(predictions)\
                .set_index(predictions.index)\
                .join(predictions)\
                .dropna()

    class Evaluation:
        errors: DataFrame
        __true_positives: float
        __false_positives: float
        __false_negatives: float
        __precision: float = None
        __recall: float = None
        __f1: float = None
        __mae_direction: float = None
        __mse_direction: float = None
        __rmse_direction: float = None
        __mae_location: float = None
        __mse_location: float = None
        __rmse_location: float = None

        def __init__(self, predictions, truths, errors) -> None:
            self.errors = errors
            self.__true_positives = len(errors)
         
            self.__false_positives = len(predictions) - self.__true_positives
            self.__false_negatives = len(truths) - self.__true_positives
        


        @property
        def mae_direction(self) -> float:
            if self.__mae_direction is None:
                self.__mae_direction = np.mean(
                    self.errors['direction_error']
                )
            return self.__mae_direction

        @property
        def mae_direction_degrees(self) -> float:
            return math.degrees(self.mae_direction)

        @property
        def mse_direction(self) -> float:
            if self.__mse_direction is None:
                self.__mse_direction = np.mean(
                    self.errors['direction_error']**2
                )
            return self.__mse_direction

        @property
        def mse_direction_degrees(self) -> float:
            return math.degrees(self.mse_direction)

        @property
        def rmse_direction(self) -> float:
            if self.__rmse_direction is None:
                self.__rmse_direction = math.sqrt(self.mse_direction)
            return self.__rmse_direction

        @property
        def rmse_direction_degrees(self) -> float:
            return math.degrees(self.rmse_direction)

  
        @property
        def mae_location(self) -> float:
            if self.__mae_location is None:
                self.__mae_location = np.mean(self.errors['location_error'])
            return self.__mae_location

        @property
        def mse_location(self) -> float:
            if self.__mse_location is None:
                self.__mse_location = np.mean(self.errors['location_error']**2)
            return self.__mse_location

        @property
        def rmse_location(self) -> float:
            if self.__rmse_location is None:
                self.__rmse_location = math.sqrt(self.mse_location)
            return self.__rmse_location\

        @property
        def precision(self):
            if self.__precision is not None:
                return self.__precision

            denominator = self.__true_positives + self.__false_positives
            self.__precision = math.nan if denominator == 0 else self.__true_positives / denominator
            return self.__precision

        @property
        def recall(self):
            if self.__recall is not None:
                return self.__recall

            denominator = self.__true_positives + self.__false_negatives
            self.__recall = math.nan if denominator == 0 else self.__true_positives / denominator
            return self.__recall

        @property
        def f1(self):
            if self.__f1 is not None:
                return self.__f1

            denominator = self.precision + self.recall
            if math.isnan(denominator) or denominator == 0:
                self.__f1 = math.nan
            else:
                self.__f1 = 2 * (self.precision * self.recall / denominator)

            return self.__f1
        
        @property
        def true_positives(self) -> float:
            return self.__true_positives

        @property
        def false_positives(self) -> float:
            return self.__false_positives

        @property
        def false_negatives(self) -> float:
            return self.__false_negatives

    def merge_predictions(self, df, distance=1):
        merge_rows = []

        for classifier, group in df.groupby(df.index.get_level_values('classifier')):
            group = group.copy()
            points = list(group['geom'])
            used = set()
            clusters = []

            for i, point in enumerate(points):
                if i in used:
                    continue
                cluster = [i]
                for j, other in enumerate(points):
                    if i != j and j not in used and point.distance(other) < distance:
                        cluster.append(j)
                used.update(cluster)
                clusters.append(cluster)

        
            for cid, cluster in enumerate(clusters):
                rows = group.iloc[cluster]

                merged = {
                    'geom': Point(
                        np.mean([pt.x for pt in rows['geom']]),
                        np.mean([pt.y for pt in rows['geom']])
                    ),
                    'classifier': classifier,
                    'cid': cid
                }

                for col in group.columns:
                    if col in ('geom', 'classifier'):
                        continue
                    elif col in ('avg_score', 'avg_speed', 'avg_heading', 'heading'):
                        merged[col] = rows[col].mean()
                    elif col in ('count', 'trip_count'):
                        merged[col] = rows[col].sum()
                    else:
                        merged[col] = rows[col].iloc[0]

                merged_index = rows.index[0]
                merge_rows.append((merged_index, merged))

        merged_df = pd.DataFrame([m for _, m in merge_rows], index=[idx for idx, _ in merge_rows])
        merged_gdf = GeoDataFrame(merged_df, geometry='geom', crs=getattr(df, 'crs', None))

        merged_gdf = merged_gdf.set_index(['cid', 'classifier'])

        return merged_gdf
    def evaluate(self) -> Evaluation:
        errors = {
            'direction_error': self.__get_direction_errors(),
            'location_error': self.__get_location_errors()
        }
        errors = DataFrame(errors, index=errors['direction_error'].index)
        return self.Evaluation(self.predictions,  self.truth_geoms, errors)
    
   
    def evaluate_label(self) -> dict:

        evaluations = {}

     
        all_predictions = self.predictions.reset_index()

     
        matched = self.data.reset_index()

    
        all_labels = all_predictions['classifier'].unique()

        for label in all_labels:
         
            preds_label = all_predictions[all_predictions['classifier'] == label]

      
            matched_label = matched[matched['classifier'] == label]

          
            truths_label = [
                g for g in self.truth_geoms
                if self.point_map[id(g)].properties.get('label_name') == label
            ]

            errors = {
                'direction_error': self.__get_angle_error(
                    matched_label['heading'], matched_label['true_heading']),
                'location_error': matched_label['geom'].distance(matched_label['true_geom'])
            }
            errors_df = DataFrame(errors, index=matched_label.index)

            evaluations[label] = self.Evaluation(
                predictions=preds_label, 
                truths=truths_label,     
                errors=errors_df         
            )

        return evaluations
    
    def __get_direction_errors(self) -> Series:
        return self.__get_angle_error(
            self.data['heading'],
            self.data['true_heading']
        )

    def __get_location_errors(self) -> Series:
        return self.data['geom'].distance(self.data['true_geom']) 

    @staticmethod
    def __get_angle_error(
        prediction: Union[Series, DataFrame, ndarray, float],
        truth: Union[Series, DataFrame, ndarray, float]
    ) -> Union[Series, float]:
        supported_types = (Series, DataFrame, ndarray, float)
        if not isinstance(prediction, supported_types):
            raise TypeError(
                f'Expected types for param "prediction" [Series, DataFrame, ndarray or float], got {type(prediction)}')
        if not isinstance(truth, supported_types):
            raise TypeError(
                f'Expected types for param "truth" [Series, DataFrame, ndarray or float], got {type(prediction)}')

        return abs(Predict.signed_angle_diff(prediction, truth))

    @staticmethod
    def __get_location_error(prediction: Point, truth: Point) -> Union[float, None]:
        if prediction is None or truth is None:
            return None

        return prediction.distance(truth)
  

    def __get_candidate_errors(self, cluster_center: Point, heading: float, candidates: List[Position]) -> Tuple[float, Position]:
        norm_location_errors = (self.__get_location_error(
            cluster_center, x.point) / 20 for x in candidates) 
       
        norm_angle_errors = (self.__get_angle_error(
            heading, x.properties['heading']) / 180 for x in candidates)
        # norm_angle_errors = (self.__get_angle_error(
        #     heading, x.properties['heading']) for x in candidates)
        norm_errors = (
            x + y for x, y in zip(norm_location_errors, norm_angle_errors))
        return zip(norm_errors, candidates)

    def __get_truth_candidates(
        self,
        classifier: str,
        cluster_center: Point,
        heading: float
    ) -> Union[None, Position]:
        query_geom = cluster_center.buffer(20)
        indices = self.truths.query(query_geom)

    
        results = [self.truth_geoms[i] for i in indices if self.truth_geoms[i].intersects(query_geom)]

   
        results = [self.point_map[id(x)] for x in results]

        results = [x for x in results if x.properties.get('label_name') == classifier]


        if len(results) == 0:
            return []

        
        results = self.__get_candidate_errors(cluster_center, heading, results)
       
        return (x[1] for x in sorted(results, key=lambda x: x[0]))

    def __get_truths_c(self, df: DataFrame) -> GeoDataFrame:
        data = zip(
            df.index.get_level_values('classifier'),
            df['geom'],
            df['heading']
        )

        visited = set()

        def get_truth(classifier: str, cluster_center: Point, heading: float) -> Union[Tuple[Point, float], None]:
            for candidate in self.__get_truth_candidates(classifier, cluster_center, heading):
           
                if candidate not in visited:
                  
                    return (candidate, candidate.properties['heading'])
                
            return (None, None)

        return GeoDataFrame(
            [
                get_truth(classifier, cluster_center, heading)
                for classifier, cluster_center, heading
                in data
            ],
            columns=['true_geom', 'true_heading'],
            crs='epsg:3044',
            geometry='true_geom',
        )

    

    def __get_truths(self, df: DataFrame) -> GeoDataFrame:
   
        data = list(zip(
            df.index.get_level_values('classifier'),
            df['geom'],
            df['heading']
        ))

        candidate_truths = []  
        candidate_points = []  
        
        for classifier, cluster_center, heading in data:
            candidates = list(self.__get_truth_candidates(classifier, cluster_center, heading))
            candidate_truths.append(candidates)
            # row_to_candidates.append([len(candidate_points) + i for i in range(len(candidates))])
            candidate_points.extend(candidates)

   
        unique_truth_points = []
        truth_point_to_idx = {}
        for truth in candidate_points:
            key = id(truth)  
            if key not in truth_point_to_idx:
               
                truth_point_to_idx[key] = len(unique_truth_points)
                unique_truth_points.append(truth)

   
        n_rows = len(data)
        n_truths = len(unique_truth_points)
        cost_matrix = np.full((n_rows, n_truths), 1e9)  

     
        for i, candidates in enumerate(candidate_truths):
            for cand in candidates:
                key = id(cand)
                j = truth_point_to_idx[key]
                cluster_center = data[i][1]
              
                dist = cluster_center.distance(cand.point)
                cost_matrix[i, j] = dist

  
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
      
        true_geoms = [None] * n_rows
        true_headings = [None] * n_rows
        assigned_truth = set()
        for i, j in zip(row_ind, col_ind):
            if cost_matrix[i, j] < 1e9:  
                truth = unique_truth_points[j]
             
                if j not in assigned_truth:
                    true_geoms[i] = truth.point
                    true_headings[i] = truth.properties['heading']
                    assigned_truth.add(j)

        return GeoDataFrame(
            {'true_geom': true_geoms, 'true_heading': true_headings},
            crs='epsg:3044',
            geometry='true_geom'
        )
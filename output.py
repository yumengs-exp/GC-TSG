import pandas as pd
import geopandas as gpd
import dataloader
from Evaluator import GoMapEvaluator
# from Split.GNN import run_graphcl_clustering, run_dbscan_clustering
from geopandas import GeoDataFrame
from Split.CL import run_cl_clustering
import os
import matplotlib.pyplot as plt
from geopandas import GeoDataFrame
import contextily as ctx
from Geolocating.jiaodian import predict_cluster_geometries
from collections import Counter
import pandas as pd
from Geolocating.Residual import train_predict
from Geolocating.Residual import train_predict

def visualize_by_class_modes(detections, predictions_by_approach, truths, point_to_truth, dataset, classes_list,approach,modes=['DT', 'TP', 'DP', 'DTP']):
    DEFAULT_CLASSES = classes_list
    os.makedirs("output", exist_ok=True)

    detections_gdf = GeoDataFrame(detections, crs='epsg:3857', geometry='geom')

    truths_df = GeoDataFrame({
        'label_name': [point_to_truth[id(p)].properties.get('label_name') for p in truths],
        'geometry': truths
    }, crs='epsg:3857')

    modes = ['DTP']
    colors = {
        'D': ('o', 'blue', 0.6),
        'T': ('x', 'green', 0.8),
        'P': ('^', 'red', 0.6)
    }

    for class_name in DEFAULT_CLASSES:
        for mode in modes:
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.set_title(f'{class_name} - {mode}', fontsize=14)

            include_d = 'D' in mode
            include_t = 'T' in mode
            include_p = 'P' in mode

            # Detections
            if include_d:
                det = detections_gdf[detections_gdf['classifier'].str.lower() == class_name.lower()]
                if not det.empty:
                    det.plot(ax=ax, marker=colors['D'][0], color=colors['D'][1], alpha=colors['D'][2],
                             label='Detection')

            # Truths
            if include_t:
                tru = truths_df[truths_df['label_name'].str.lower() == class_name.lower()]
                if not tru.empty:
                    tru.plot(ax=ax, marker=colors['T'][0], color=colors['T'][1], alpha=colors['T'][2],
                             label='Ground Truth')

            # Predictions
            if include_p:
                pred_df = predictions_by_approach
                pred_df = pred_df.reset_index()
                pred_class = pred_df[pred_df['classifier'].str.lower() == class_name.lower()]
                if not pred_class.empty:
                    geo_pred = GeoDataFrame(pred_class, crs='epsg:3857', geometry='geom')
                    geo_pred.plot(ax=ax, marker=colors['P'][0], color=colors['P'][1], alpha=colors['P'][2],
                                  label=f'Prediction: MLP')

            ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, crs='epsg:3857')
            ax.legend()
            plt.tight_layout()

            suffix = '' if mode == 'DTP' else f'_{mode}'
            filename = f"output/{dataset}/image/{class_name.replace(':', '_')}{suffix}_{approach}.png"
            plt.savefig(filename, dpi=300)
            plt.close()
            print(f"✅ Save: {filename}")


def print_evaluation_metrics_by_class(evaluator: GoMapEvaluator, classes_list):
    DEFAULT_CLASSES = classes_list

    label_evaluations = evaluator.evaluate_label()

    for label in sorted(DEFAULT_CLASSES):
        evaluation = label_evaluations.get(label)

        print(f"\n📦 Evaluation for label: {label}")
        print("-------------------------------")

        if evaluation is None or evaluation.errors.empty:
            print("❌ No predictions or matches found for this label.")
            continue

        print("📊 Evaluation Metrics")
        print(f"✅ Precision        : {evaluation.precision:.4f}")
        print(f"✅ Recall           : {evaluation.recall:.4f}")
        print(f"✅ F1 Score         : {evaluation.f1:.4f}")

        # 新增：打印 TP, FP, FN
        print()
        print(f"TP (True Positives) : {evaluation.true_positives}")
        print(f"FP (False Positives): {evaluation.false_positives}")
        print(f"FN (False Negatives): {evaluation.false_negatives}")

        print()
        print("📍 Location Error")
        print(f" - MAE (Mean Abs)  : {evaluation.mae_location:.4f} meters")
        print(f" - MSE (Mean Sq)   : {evaluation.mse_location:.4f} m²")
        print(f" - RMSE (Root MSE) : {evaluation.rmse_location:.4f} meters")


def print_evaluation_metrics(evaluator: GoMapEvaluator):
    evaluation = evaluator.evaluate()

    print("📊 Evaluation Metrics")
    print("----------------------")
    print(f"✅ Precision        : {evaluation.precision:.4f}")
    print(f"✅ Recall           : {evaluation.recall:.4f}")
    print(f"✅ F1 Score         : {evaluation.f1:.4f}")

 
    print()
    print(f"TP (True Positives) : {evaluation.true_positives}")
    print(f"FP (False Positives): {evaluation.false_positives}")
    print(f"FN (False Negatives): {evaluation.false_negatives}")

    print()
    print("📍 Location Error")
    print(f" - MAE (Mean Abs)  : {evaluation.mae_location:.4f} meters")
    print(f" - MSE (Mean Sq)   : {evaluation.mse_location:.4f} m²")
    print(f" - RMSE (Root MSE) : {evaluation.rmse_location:.4f} meters")

def get_evaluation_metrics(evaluator: GoMapEvaluator):
    evaluation = evaluator.evaluate()
    return evaluation.f1, evaluation.mae_location


import numpy as np
import pandas as pd
import geopandas as gpd


X_SPLIT = {
    "b11": 1112562.4837953418,
    "c55:50": 1110743.227038553,
    "c55:60": 1112563.8041722607,
    "c61": 1112450.6594250074,
    "d15.3": 1112556.2448023255,
}

def _get_classifier_series(gdf: gpd.GeoDataFrame, classifier_name="classifier") -> pd.Series:
    if classifier_name in gdf.columns:
        return gdf[classifier_name].astype(str).str.strip()
    idx = gdf.index
    if isinstance(idx, pd.MultiIndex):
        if classifier_name in idx.names:
            return pd.Series(idx.get_level_values(classifier_name).astype(str).str.strip(), index=gdf.index, name=classifier_name)
        
        return pd.Series(idx.get_level_values(-1).astype(str).str.strip(), index=gdf.index, name=classifier_name)

    return pd.Series(idx.astype(str), index=gdf.index, name=classifier_name)

def filter_gf1_by_xsplit(gf1: gpd.GeoDataFrame, keep_inclusive=False) -> gpd.GeoDataFrame:

    if gf1.geometry.name != "true_geom":
        gf1 = gf1.set_geometry("true_geom")

    cls = _get_classifier_series(gf1, "classifier")
    thresholds = cls.map(X_SPLIT)  
    x = gf1.geometry.x
    cond = thresholds.notna() & (x >= thresholds if keep_inclusive else x > thresholds)

    return gf1.loc[cond].copy()

def geom_ids(
    gdf: gpd.GeoDataFrame,
    tol_decimals: int | None = None,
):

    geom_col = gdf.geometry.name
    geoms = gdf[geom_col]


    def is_point_or_none(g):
        if g is None:
            return True
        try:
            return g.geom_type == "Point"
        except Exception:
            return False

    if not geoms.apply(is_point_or_none).all():
        raise ValueError("true_geom MUST BE POINT)")

    if tol_decimals is None:

        return geoms.apply(lambda g: None if g is None else g.wkb_hex)
    else:
   
        return geoms.apply(
            lambda g: None if g is None else (round(g.x, tol_decimals), round(g.y, tol_decimals))
        )

def compare_gf2_vs_gf1_true_geom(
    gf1: gpd.GeoDataFrame,
    gf2: gpd.GeoDataFrame,
    tol_decimals: int | None = None, 
):
    gf1_filtered = filter_gf1_by_xsplit(gf1, keep_inclusive=False)
  
    if gf1_filtered.geometry.name is None or gf2.geometry.name is None:
        raise ValueError("gf1_filtered ERROR")

    s1 = geom_ids(gf1_filtered, tol_decimals=tol_decimals)
    s2 = geom_ids(gf2,          tol_decimals=tol_decimals)

    set1 = set(s1.dropna())
    set2 = set(s2.dropna())

    # 新增：gf2 中有但 gf1_filtered 中没有
    new_mask = ~s2.isna() & ~s2.isin(set1)
    # 删除：gf1_filtered 中有但 gf2 中没有
    del_mask = ~s1.isna() & ~s1.isin(set2)

    added_in_gf2 = gf2.loc[new_mask].copy()
    removed_from_gf2 = gf1_filtered.loc[del_mask].copy()

    return added_in_gf2, removed_from_gf2

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx


X_SPLIT = {
    "b11": 1112562.4837953418,
    "c55:50": 1110743.227038553,
    "c55:60": 1112563.8041722607,
    "c61": 1112450.6594250074,
    "d15.3": 1112556.2448023255,
    "c62": 1112624.8552007228,
    "e56": 1112582.029461274,
    "n42.3": 1112562.5102727339,
    "c56:60": 1112616.915358799,
    "e55": 1112554.5470830791,
    "e33.1": 1110111.4262186142,
    "d11.3": 1112601.839159253
}



def _get_classifier_series(gdf: gpd.GeoDataFrame, name="classifier") -> pd.Series:

    if name in gdf.columns:
        return gdf[name].astype(str).str.strip()
    idx = gdf.index
    if isinstance(idx, pd.MultiIndex):
        if name in idx.names:
            return pd.Series(idx.get_level_values(name).astype(str).str.strip(),
                             index=gdf.index, name=name)

        return pd.Series(idx.get_level_values(-1).astype(str).str.strip(),
                         index=gdf.index, name=name)
    return pd.Series(idx.astype(str), index=gdf.index, name=name)

def _wkb_id(g):
    return None if g is None else getattr(g, "wkb_hex", None)

def _truth_status_by_hits(truths_gdf: gpd.GeoDataFrame,
                          set_hit1: set, set_hit2: set) -> dict:
    """按 (both, new_in_gf2, missing_in_gf2, both_missing) 划分 truth 行"""
    ids = truths_gdf.geometry.apply(_wkb_id)
    both      = truths_gdf[ids.apply(lambda x: x in set_hit1 and x in set_hit2)]
    new_gf2   = truths_gdf[ids.apply(lambda x: x not in set_hit1 and x in set_hit2)]
    miss_gf2  = truths_gdf[ids.apply(lambda x: x in set_hit1 and x not in set_hit2)]
    both_miss = truths_gdf[ids.apply(lambda x: x not in set_hit1 and x not in set_hit2)]
    return {"both": both, "new_gf2": new_gf2, "miss_gf2": miss_gf2, "both_miss": both_miss}

def _filter_gf1_by_xsplit(gf1: gpd.GeoDataFrame, xsplit: dict) -> gpd.GeoDataFrame:

    if "true_geom" not in gf1.columns:
        raise ValueError("gf1 NEED TO INCLUDE 'true_geom'（Point）")
    g = gf1.set_geometry("true_geom")
    cls = _get_classifier_series(g, "classifier")
    thr = cls.map(xsplit) 
    cond = thr.notna() & (g.geometry.x > thr)
    return gf1.loc[cond].copy()

def _filter_truths_by_xsplit(truths_df: gpd.GeoDataFrame, class_name: str, xsplit: dict):

    t = truths_df[truths_df["label_name"].str.lower() == class_name.lower()].copy()
    if class_name in xsplit:
        t = t[t.geometry.x > xsplit[class_name]]
    return t

def _filter_truths_by_class(truths_df: gpd.GeoDataFrame, class_name: str):

    t = truths_df[truths_df["label_name"].str.lower() == class_name.lower()].copy()
    return t

def _filter_by_xsplit_geom(gdf: gpd.GeoDataFrame, xsplit: dict, geom_col="geom") -> gpd.GeoDataFrame:
   
    if geom_col not in gdf.columns:
        raise ValueError(f"{geom_col} does not exist")
    g = gdf.set_geometry(geom_col)
    cls = _get_classifier_series(g, "classifier")
    thr = cls.map(xsplit)
    cond = thr.notna() & (g.geometry.x > thr)
    return gdf.loc[cond].copy()


def visualize_diffs_by_class(
    gf1: gpd.GeoDataFrame,
    gf2: gpd.GeoDataFrame,
    prediction1: gpd.GeoDataFrame,   
    prediction2: gpd.GeoDataFrame,   
    truths,                         
    point_to_truth,
    classes_list,
    dataset: str,
    approach: str,
    xsplit_map: dict = None,
    out_dir: str = "output",
    fixed_map_width_m: float = 2000.0 
):
    os.makedirs(f"{out_dir}/{dataset}/image", exist_ok=True)
    xsplit_map = xsplit_map or X_SPLIT


    truths_df = gpd.GeoDataFrame({
        "label_name": [point_to_truth[id(p)].properties.get("label_name") for p in truths],
        "geometry": truths
    }, crs="epsg:3857")


    gf1_f = _filter_gf1_by_xsplit(gf1, xsplit_map)

   
    pred_marker = "^"
    pred_colors = {"prediction1": "#1f77b4", "prediction2": "#ff7f0e"}

 
    truth_marker = "x"
    truth_colors = {
        "both": "#2ca02c",
        "new_gf2": "#830864",
        "miss_gf2": "#d62728",
        "both_miss": "#7f7f7f",
    }

    for class_name in classes_list:
  
        s1 = _get_classifier_series(gf1_f, "classifier")
        s2 = _get_classifier_series(gf2,   "classifier")
        gf1_c = gf1_f[s1.str.lower() == class_name.lower()].copy()
        gf2_c = gf2[s2.str.lower() == class_name.lower()].copy()

     
        sp1 = _get_classifier_series(prediction1, "classifier")
        sp2 = _get_classifier_series(prediction2, "classifier")
        pred1_c = prediction1[sp1.str.lower() == class_name.lower()].copy()
        pred2_c = prediction2[sp2.str.lower() == class_name.lower()].copy()
        if not pred1_c.empty:
            pred1_c = _filter_by_xsplit_geom(pred1_c, xsplit_map, geom_col="geom")

       
        truths_c = _filter_truths_by_xsplit(truths_df, class_name, xsplit_map)

    
        set_hit1 = set(gf1_c["true_geom"].dropna().apply(_wkb_id))
        set_hit2 = set(gf2_c["true_geom"].dropna().apply(_wkb_id))

        truth_bins = _truth_status_by_hits(truths_c, set_hit1, set_hit2)

        if all(df.empty for df in truth_bins.values()):
            continue

    
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_title(f"{class_name} – preds & truth diffs ({approach})", fontsize=14)

    
        if not pred1_c.empty:
            gpred1 = gpd.GeoDataFrame(pred1_c.copy(), geometry="geom", crs="epsg:3857")
            gpred1.plot(ax=ax, marker=pred_marker, color=pred_colors["prediction1"], alpha=0.8, label="Prediction1")
        if not pred2_c.empty:
            gpred2 = gpd.GeoDataFrame(pred2_c.copy(), geometry="geom", crs="epsg:3857")
            gpred2.plot(ax=ax, marker=pred_marker, color=pred_colors["prediction2"], alpha=0.8, label="Prediction2")


        if not truth_bins["both"].empty:
            truth_bins["both"].plot(ax=ax, marker=truth_marker, color=truth_colors["both"], alpha=0.9, label="Truth (both hit)")
        if not truth_bins["new_gf2"].empty:
            truth_bins["new_gf2"].plot(ax=ax, marker=truth_marker, color=truth_colors["new_gf2"], alpha=0.9, label="Truth (new in gf2)")
        if not truth_bins["miss_gf2"].empty:
            truth_bins["miss_gf2"].plot(ax=ax, marker=truth_marker, color=truth_colors["miss_gf2"], alpha=0.9, label="Truth (missing in gf2)")
        if not truth_bins["both_miss"].empty:
            truth_bins["both_miss"].plot(ax=ax, marker=truth_marker, color=truth_colors["both_miss"], alpha=0.7, label="Truth (both missing)")

        PAD_RATIO = 0.10
        MIN_WIDTH = 200.0
        MIN_HEIGHT = 200.0

        def _collect_xy():
            xs, ys = [], []
            if not truths_c.empty:
                xs.extend(truths_c.geometry.x.tolist())
                ys.extend(truths_c.geometry.y.tolist())
            if not pred1_c.empty:
                xs.extend(pred1_c["geom"].x.tolist())
                ys.extend(pred1_c["geom"].y.tolist())
            if not pred2_c.empty:
                xs.extend(pred2_c["geom"].x.tolist())
                ys.extend(pred2_c["geom"].y.tolist())
            return xs, ys

        xs, ys = _collect_xy()

        if len(xs) >= 1 and len(ys) >= 1:
            minx, maxx = min(xs), max(xs)
            miny, maxy = min(ys), max(ys)
            width = maxx - minx
            height = maxy - miny
            if width == 0:  width = 1.0
            if height == 0: height = 1.0
            width = max(width, MIN_WIDTH)
            height = max(height, MIN_HEIGHT)
            cx = (minx + maxx) / 2.0
            cy = (miny + maxy) / 2.0
            half_w = width / 2.0
            half_h = height / 2.0
            pad_w = width * PAD_RATIO
            pad_h = height * PAD_RATIO
            ax.set_xlim(cx - half_w - pad_w, cx + half_w + pad_w)
            ax.set_ylim(cy - half_h - pad_h, cy + half_h + pad_h)
        else:
            ax.set_xlim(0, 100)
            ax.set_ylim(0, 100)

      
        try:
            ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, crs="epsg:3857")
        except Exception:
            pass

        ax.legend()
        plt.tight_layout()

 
        filename = f"{out_dir}/{dataset}/image/{class_name.replace(':','_')}_diff_{approach}.png"
        plt.savefig(filename, dpi=300)
        plt.close()
        print(f"Save: {filename} | "
              f"truths: total={len(truths_c)}, both={len(truth_bins['both'])}, "
              f"new_gf2={len(truth_bins['new_gf2'])}, miss_gf2={len(truth_bins['miss_gf2'])}, "
              f"both_miss={len(truth_bins['both_miss'])}")
        




def visualize_diffs_by_class_skipfilter(
    gf1: gpd.GeoDataFrame,
    gf2: gpd.GeoDataFrame,
    prediction1: gpd.GeoDataFrame,   # 🔧 CHANGED: 新增
    prediction2: gpd.GeoDataFrame,   # 🔧 CHANGED: 新增
    truths,                          # Iterable[Point], EPSG:3857
    point_to_truth,
    classes_list,
    dataset: str,
    approach: str,
    xsplit_map: dict = None,
    out_dir: str = "output",
    fixed_map_width_m: float = 2000.0 # 🔧 CHANGED: 固定地图宽度（米）
):
    os.makedirs(f"{out_dir}/{dataset}/image", exist_ok=True)
    # xsplit_map = xsplit_map or X_SPLIT

    # truths gdf
    truths_df = gpd.GeoDataFrame({
        "label_name": [point_to_truth[id(p)].properties.get("label_name") for p in truths],
        "geometry": truths
    }, crs="epsg:3857")

    # 仅过滤 gf1（用于 truth 命中判定）
    gf1_f = gf1

    # 预测点的样式（同形状，不同颜色）
    pred_marker = "^"
    pred_colors = {"prediction1": "#1f77b4", "prediction2": "#ff7f0e"}  # 蓝/橙

    # truths 的着色：四种状态
    truth_marker = "x"
    truth_colors = {
        "both": "#2ca02c",
        "new_gf2": "#830864",
        "miss_gf2": "#d62728",
        "both_miss": "#7f7f7f",
    }

    for class_name in classes_list:
        # —— 该类的子集 —— 
        s1 = _get_classifier_series(gf1_f, "classifier")
        s2 = _get_classifier_series(gf2,   "classifier")
        gf1_c = gf1_f[s1.str.lower() == class_name.lower()].copy()
        gf2_c = gf2[s2.str.lower() == class_name.lower()].copy()

        # 🔧 CHANGED: 预测点来源改为 prediction1/2
        sp1 = _get_classifier_series(prediction1, "classifier")
        sp2 = _get_classifier_series(prediction2, "classifier")
        pred1_c = prediction1[sp1.str.lower() == class_name.lower()].copy()
        pred2_c = prediction2[sp2.str.lower() == class_name.lower()].copy()
        if not pred1_c.empty:
            pred1_c = pred1_c

        # —— 该类的 truths（按 x_split 过滤）——
        truths_c = _filter_truths_by_class(truths_df, class_name)

        # —— truth 命中集合（仍然基于 gf1/gf2 的 true_geom）——
        set_hit1 = set(gf1_c["true_geom"].dropna().apply(_wkb_id))
        set_hit2 = set(gf2_c["true_geom"].dropna().apply(_wkb_id))

        truth_bins = _truth_status_by_hits(truths_c, set_hit1, set_hit2)

        # 🔧 CHANGED: 若四类 truths 都为空，跳过
        if all(df.empty for df in truth_bins.values()):
            continue

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_title(f"{class_name} – preds & truth diffs ({approach})", fontsize=14)

        # 画 prediction1 / prediction2（geom）
        if not pred1_c.empty:
            gpred1 = gpd.GeoDataFrame(pred1_c.copy(), geometry="geom", crs="epsg:3857")
            gpred1.plot(ax=ax, marker=pred_marker, color=pred_colors["prediction1"], alpha=0.8, label="Prediction1")
        if not pred2_c.empty:
            gpred2 = gpd.GeoDataFrame(pred2_c.copy(), geometry="geom", crs="epsg:3857")
            gpred2.plot(ax=ax, marker=pred_marker, color=pred_colors["prediction2"], alpha=0.8, label="Prediction2")

        # truths：四种命中状态
        if not truth_bins["both"].empty:
            truth_bins["both"].plot(ax=ax, marker=truth_marker, color=truth_colors["both"], alpha=0.9, label="Truth (both hit)")
        if not truth_bins["new_gf2"].empty:
            truth_bins["new_gf2"].plot(ax=ax, marker=truth_marker, color=truth_colors["new_gf2"], alpha=0.9, label="Truth (new in gf2)")
        if not truth_bins["miss_gf2"].empty:
            truth_bins["miss_gf2"].plot(ax=ax, marker=truth_marker, color=truth_colors["miss_gf2"], alpha=0.9, label="Truth (missing in gf2)")
        if not truth_bins["both_miss"].empty:
            truth_bins["both_miss"].plot(ax=ax, marker=truth_marker, color=truth_colors["both_miss"], alpha=0.7, label="Truth (both missing)")

        # ===== 自适应地图范围（包含 truths + pred1 + pred2），并最小宽/高+边距 =====
        PAD_RATIO = 0.10
        MIN_WIDTH = 200.0
        MIN_HEIGHT = 200.0

        def _collect_xy():
            xs, ys = [], []
            if not truths_c.empty:
                xs.extend(truths_c.geometry.x.tolist())
                ys.extend(truths_c.geometry.y.tolist())
            if not pred1_c.empty:
                xs.extend(pred1_c["geom"].x.tolist())
                ys.extend(pred1_c["geom"].y.tolist())
            if not pred2_c.empty:
                xs.extend(pred2_c["geom"].x.tolist())
                ys.extend(pred2_c["geom"].y.tolist())
            return xs, ys

        xs, ys = _collect_xy()

        if len(xs) >= 1 and len(ys) >= 1:
            minx, maxx = min(xs), max(xs)
            miny, maxy = min(ys), max(ys)
            width = maxx - minx
            height = maxy - miny
            if width == 0:  width = 1.0
            if height == 0: height = 1.0
            width = max(width, MIN_WIDTH)
            height = max(height, MIN_HEIGHT)
            cx = (minx + maxx) / 2.0
            cy = (miny + maxy) / 2.0
            half_w = width / 2.0
            half_h = height / 2.0
            pad_w = width * PAD_RATIO
            pad_h = height * PAD_RATIO
            ax.set_xlim(cx - half_w - pad_w, cx + half_w + pad_w)
            ax.set_ylim(cy - half_h - pad_h, cy + half_h + pad_h)
        else:
            ax.set_xlim(0, 100)
            ax.set_ylim(0, 100)

        # 底图
        try:
            ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, crs="epsg:3857")
        except Exception:
            pass


        ax.legend()
        plt.tight_layout()

        # 保存
        filename = f"{out_dir}/{dataset}/image/{class_name.replace(':','_')}_diff_{approach}.png"
        plt.savefig(filename, dpi=300)
        plt.close()
        print(f"Save: {filename} | "
              f"truths: total={len(truths_c)}, both={len(truth_bins['both'])}, "
              f"new_gf2={len(truth_bins['new_gf2'])}, miss_gf2={len(truth_bins['miss_gf2'])}, "
              f"both_miss={len(truth_bins['both_miss'])}")
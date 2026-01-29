import time
import numpy as np
import torch
from shapely.geometry import Point
import pandas as pd
from shapely.geometry import Point
from geopandas import GeoDataFrame
from shapely.strtree import STRtree

class ResidualRefineNet(torch.nn.Module):
    def __init__(self, feat_dim, hidden=16, max_offset=5.0):
        super().__init__()
        self.max_offset = max_offset
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(feat_dim, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, 2)
        )

    def forward(self, features):
        offset = torch.tanh(self.mlp(features)) * self.max_offset
        return offset



def chamfer_distance_per_class(pred, gt, penalty_factor=1.0, dup_thresh=2.0):
    if pred.shape[0] == 0 or gt.shape[0] == 0:
        return torch.tensor(0.0, device=pred.device), torch.tensor(0.0, device=pred.device)
    dmat = torch.cdist(pred, gt)  
    pred2gt = dmat.min(dim=1)[0]
    gt2pred = dmat.min(dim=0)[0]
    cd_loss = pred2gt.mean() + gt2pred.mean()

    hits = (dmat < dup_thresh).float().sum(dim=0) 
    penalty = ((hits - 1.0).clamp(min=0)).sum() / (gt.shape[0] + 1e-6)
    penalty = penalty_factor * penalty

    return cd_loss, penalty

def standardize_feats(feats):
    mean = feats.mean(axis=0, keepdims=True)
    std = feats.std(axis=0, keepdims=True) + 1e-8
    return (feats - mean) / std, mean, std
from scipy.optimize import linear_sum_assignment



def grid_split(all_pred_coords, all_truth_coords, grid_size=100, train_ratio=0.8, random_seed=42):

    np.random.seed(random_seed)
    xy = all_pred_coords.cpu().numpy()
    truth_xy = all_truth_coords.cpu().numpy()


    min_x, max_x = xy[:,0].min(), xy[:,0].max()
    min_y, max_y = xy[:,1].min(), xy[:,1].max()

    def get_grid_ids(xy):
        x_ids = ((xy[:,0] - min_x) // grid_size).astype(int)
        y_ids = ((xy[:,1] - min_y) // grid_size).astype(int)
        return list(zip(x_ids, y_ids))

    pred_grid_ids = get_grid_ids(xy)
    truth_grid_ids = get_grid_ids(truth_xy)


    unique_grids = sorted(list(set(pred_grid_ids)))
    np.random.shuffle(unique_grids)
    n_train = int(len(unique_grids) * train_ratio)
    train_grids = set(unique_grids[:n_train])
    test_grids  = set(unique_grids[n_train:])

  
    train_idx = np.array([i for i, g in enumerate(pred_grid_ids) if g in train_grids])
    test_idx  = np.array([i for i, g in enumerate(pred_grid_ids) if g in test_grids])

    train_truth_idx = np.array([i for i, g in enumerate(truth_grid_ids) if g in train_grids])
    test_truth_idx  = np.array([i for i, g in enumerate(truth_grid_ids) if g in test_grids])

    return train_idx, test_idx, train_truth_idx, test_truth_idx

def train_predict(
        result: dict,
        truth_points,
        point_to_truth,
        search_tree,
        batch_size=128,
        top_k=5,
        epochs=1000,
        device='cuda',
        penalty_factor=0.1, 
        penalty_thresh=2.0,
        grid_size=100,
        train_ratio=0.8,
        random_seed=11,
        lr=1e-2,
        max_offset=1
    ):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    all_pred_coords, all_truth_coords, all_pred_feats, all_pred_cids, all_pred_types = [], [], [], [], []
    class_names = list(result.keys())


    for cls in class_names:
        preds = result[cls]
        for cid, group in preds.groupby('cid'):
            coords = np.array([[geom.x, geom.y] for geom in group['geom']])
            center = coords.mean(axis=0)
            bboxes = group[['width', 'height']].to_numpy()
            areas = bboxes[:, 0] * bboxes[:, 1]
            max_bbox_idx = areas.argmax()
            max_bbox_vec = coords[max_bbox_idx] - center
            if len(coords) > 2:
                dists = np.linalg.norm(coords - center, axis=1)
                densest_idx = dists.argmin()
                densest_vec = coords[densest_idx] - center
            else:
                densest_vec = np.zeros(2)
            heading = group['heading'].to_numpy()
            heading_complex = np.exp(1j * heading)
            heading_mean = heading_complex.mean()
            heading_real, heading_imag = np.real(heading_mean), np.imag(heading_mean)
            main_vec = max_bbox_vec if np.linalg.norm(max_bbox_vec) > 1e-3 else densest_vec
            if np.linalg.norm(main_vec) > 1e-3:
                main_dir = main_vec / (np.linalg.norm(main_vec) + 1e-8)
            else:
                main_dir = np.zeros(2)
            feat = np.concatenate([
                center, max_bbox_vec, main_dir, [heading_real, heading_imag],
            ]) 
            all_pred_coords.append(center)
            all_pred_feats.append(feat)
            all_pred_cids.append(cid)
            all_pred_types.append(cls)

    all_pred_feats = np.stack(all_pred_feats)
    all_pred_feats, feat_mean, feat_std = standardize_feats(all_pred_feats)
    all_pred_coords = np.stack(all_pred_coords)

    all_truth_coords = []
    all_truth_types = []
    for tp in truth_points:
        cls = point_to_truth[id(tp)].properties.get('label_name')
        xy = np.array([tp.x, tp.y])
        all_truth_coords.append(xy)
        all_truth_types.append(cls)
    all_truth_coords = np.stack(all_truth_coords)

    all_pred_feats = torch.tensor(all_pred_feats, dtype=torch.float32)
    all_pred_coords = torch.tensor(all_pred_coords, dtype=torch.float32)
    all_truth_coords = torch.tensor(all_truth_coords, dtype=torch.float32)
    all_pred_types = np.array(all_pred_types)
    all_truth_types = np.array(all_truth_types)


    stats = {}
    skip_classes = set()
    for cls in np.unique(all_pred_types):
        pred_mask = (all_pred_types == cls)
        truth_mask = (all_truth_types == cls)
        preds = all_pred_coords[pred_mask]
        truths = all_truth_coords[truth_mask]
        if len(preds) == 0 or len(truths) == 0:
            continue
        
        dmat = torch.cdist(preds, truths)   # [N, M]

      
        row_ind, col_ind = linear_sum_assignment(dmat.cpu().numpy())

    
        matched_dists = dmat[row_ind, col_ind]

    
        min_dists = torch.full((dmat.size(0),), float(1000), device=dmat.device)
        min_dists[row_ind] = matched_dists

        stats[cls] = min_dists.cpu().numpy()
        mean_dist = min_dists.mean().item()
        print(f"=== cls {cls} ===")
        print(f"prediction: {len(preds)}, real sign: {len(truths)}")
        print(f"mean_dist: {mean_dist:.2f} m,  median: {min_dists.median().item():.2f}m, max: {min_dists.max().item():.2f}m, min: {min_dists.min().item():.2f}m")
        print(f"5%: {np.percentile(stats[cls], 5):.2f}, 95%: {np.percentile(stats[cls], 95):.2f}")
        if mean_dist < 10.0:
            skip_classes.add(cls)

  
    train_idx, test_idx, train_truth_idx, test_truth_idx = grid_split(
        all_pred_coords, all_truth_coords, grid_size=grid_size,
        train_ratio=train_ratio, random_seed=random_seed
    )


    feat_dim = all_pred_feats.shape[1]
    net = ResidualRefineNet(feat_dim, max_offset=max_offset).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    for epoch in range(epochs):
        net.train()
        opt.zero_grad()
        loss_all = 0
        penalty_all = 0
        count = 0

        for cls in class_names:
            if cls in skip_classes:
                continue  
            pred_mask = (all_pred_types[train_idx] == cls)
            truth_mask = (all_truth_types[train_truth_idx] == cls)
            if pred_mask.sum() == 0 or truth_mask.sum() == 0:
                continue
            pred_feats = all_pred_feats[train_idx][pred_mask].to(device)
            pred_centers = all_pred_coords[train_idx][pred_mask].to(device)
            pred_offsets = net(pred_feats)
            refined = pred_centers + pred_offsets
            gt = all_truth_coords[train_truth_idx][truth_mask].to(device)
            offset_penalty = torch.relu(pred_offsets.norm(dim=1) - 3.0).mean()
            loss, penalty = chamfer_distance_per_class(
                refined, gt, penalty_factor=penalty_factor, dup_thresh=penalty_thresh
            )
            loss_all += loss
            penalty_all += penalty
            count += 1

        if count == 0:
            continue
        loss_all = loss_all / count

        loss_all.backward()
        opt.step()
        if epoch % 50 == 0 or epoch == epochs - 1:
            print(f'Epoch {epoch}: CD Loss={loss_all.item():.4f} | Penalty={penalty_all.item() / max(count, 1):.4f}')

    net.eval()
    results = []
    no_results = []
    starttime = time.time()
    with torch.no_grad():
        for cls in class_names:
            pred_mask = (all_pred_types[test_idx] == cls)
            if pred_mask.sum() == 0:
                continue
            test_centers = all_pred_coords[test_idx][pred_mask].to(device)
            indices = test_idx[pred_mask]
            if cls in skip_classes:
                x_pred = test_centers[:, 0].cpu().numpy()
                y_pred = test_centers[:, 1].cpu().numpy()

                x_pred_no = test_centers[:, 0].cpu().numpy()
                y_pred_no = test_centers[:, 1].cpu().numpy()
            else:
                test_feats = all_pred_feats[test_idx][pred_mask].to(device)
                test_offsets = net(test_feats)
                refined_test = test_centers + test_offsets
                x_pred = refined_test[:, 0].cpu().numpy()
                y_pred = refined_test[:, 1].cpu().numpy()

                x_pred_no = test_centers[:, 0].cpu().numpy()
                y_pred_no = test_centers[:, 1].cpu().numpy()
            for i, idx in enumerate(indices):
                results.append({
                    "cid": all_pred_cids[idx],
                    "classifier": cls,
                    "geom": Point(x_pred[i], y_pred[i]),
                    "avg_score": 0.0,
                    "avg_speed": 0,
                    "avg_heading": 0.0,
                    "heading": 0.0,
                    "count": 0,
                    "trip_count": 0
                })

                no_results.append({
                    "cid": all_pred_cids[idx],
                    "classifier": cls,
                    "geom": Point(x_pred_no[i], y_pred_no[i]),
                    "avg_score": 0.0,
                    "avg_speed": 0,
                    "avg_heading": 0.0,
                    "heading": 0.0,
                    "count": 0,
                    "trip_count": 0
                })
    endtime = time.time()
    print(f"predict time: {endtime - starttime:.2f} 秒")

    df_test = pd.DataFrame(results)
    gdf_test = GeoDataFrame(df_test, geometry="geom", crs="EPSG:4326")
    gdf_test.set_index(["cid", "classifier"], inplace=True)

    df_test_no = pd.DataFrame(no_results)
    gdf_test_no = GeoDataFrame(df_test_no, geometry="geom", crs="EPSG:4326")
    gdf_test_no.set_index(["cid", "classifier"], inplace=True)


    right_truth_points = [truth_points[idx] for idx in test_truth_idx]
    right_point_to_truth = {id(pt): point_to_truth[id(pt)] for pt in right_truth_points}
    right_search_tree = STRtree(right_truth_points) if right_truth_points else None


    train_results = []
    with torch.no_grad():
        for cls in class_names:
            pred_mask = (all_pred_types[train_idx] == cls)
            if pred_mask.sum() == 0:
                continue
            train_centers = all_pred_coords[train_idx][pred_mask].to(device)
            indices = train_idx[pred_mask]
            if cls in skip_classes:
                x_pred = train_centers[:, 0].cpu().numpy()
                y_pred = train_centers[:, 1].cpu().numpy()
            else:
                train_feats = all_pred_feats[train_idx][pred_mask].to(device)
                train_offsets = net(train_feats)
                refined_train = train_centers + train_offsets
                x_pred = refined_train[:, 0].cpu().numpy()
                y_pred = refined_train[:, 1].cpu().numpy()
            for i, idx in enumerate(indices):
                train_results.append({
                    "cid": all_pred_cids[idx],
                    "classifier": cls,
                    "geom": Point(x_pred[i], y_pred[i]),
                    "avg_score": 0.0,
                    "avg_speed": 0,
                    "avg_heading": 0.0,
                    "heading": 0.0,
                    "count": 0,
                    "trip_count": 0
                })
    df_train = pd.DataFrame(train_results)
    gdf_train = GeoDataFrame(df_train, geometry="geom", crs="EPSG:4326")
    gdf_train.set_index(["cid", "classifier"], inplace=True)
    left_truth_points = [truth_points[idx] for idx in train_truth_idx]
    left_point_to_truth = {id(pt): point_to_truth[id(pt)] for pt in left_truth_points}
    left_search_tree = STRtree(left_truth_points) if left_truth_points else None



    return gdf_train, left_truth_points, left_point_to_truth, left_search_tree, \
        gdf_test, right_truth_points, right_point_to_truth, right_search_tree,\
        gdf_test_no, right_truth_points, right_point_to_truth, right_search_tree

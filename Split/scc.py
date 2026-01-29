import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from torch.utils.data import DataLoader, TensorDataset
import hdbscan
import math
import os
import pickle
from shapely.geometry import Point
from shapely.strtree import STRtree
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics.pairwise import cosine_similarity



def build_data(df, save_path=None):
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    import numpy as np
    import torch

    features = []
    positions = []

    for _, row in df.iterrows():
        try:
            x_pos = getattr(row['geom'], 'x', 0.0) if pd.notnull(row['geom']) else 0.0
            y_pos = getattr(row['geom'], 'y', 0.0) if pd.notnull(row['geom']) else 0.0
            heading_cos = float(np.cos(row['heading'])) if pd.notnull(row['heading']) else 1.0
            heading_sin = float(np.sin(row['heading'])) if pd.notnull(row['heading']) else 0.0
            norm_width = row['width'] / row['img_width'] if pd.notnull(row['width']) and pd.notnull(row['img_width']) and row['img_width'] != 0 else 0.0
            norm_height = row['height'] / row['img_height'] if pd.notnull(row['height']) and pd.notnull(row['img_height']) and row['img_height'] != 0 else 0.0
            feat = [x_pos, y_pos, heading_cos, heading_sin, norm_width, norm_height]
        except Exception as e:
            print(f"[WARN] Feature extract error: {e}")
            continue

        features.append(feat)
        positions.append([x_pos, y_pos])

    if len(features) == 0:
        raise ValueError("No valid features extracted from input DataFrame.")

    x = torch.tensor(features, dtype=torch.float)  # [N,6]
    if torch.isnan(x).any():
        raise ValueError("Feature tensor contains NaN values.")

 
    x_normed = x / (x.sum(dim=1, keepdim=True) + 1e-8)
    return x_normed, torch.tensor(positions, dtype=torch.float)

def augment(x, noise_std=0.01):
    noise = torch.randn_like(x) * noise_std
    return x + noise

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
    def forward(self, x):
        x = self.net(x)
        return F.normalize(x, dim=1)

def nt_xent_loss(z1, z2, temperature=0.5):
    N = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)
    sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)
    labels = torch.arange(N, device=z.device)
    labels = torch.cat([labels, labels], dim=0)
    logits = sim / temperature
    return F.cross_entropy(logits, labels)

def train_cl(data, input_dim, hidden_dim=16, epochs=3000, lr=1e-1, batch_size=512):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MLP(input_dim, hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(TensorDataset(data), batch_size=batch_size, shuffle=True, drop_last=False)
    for epoch in range(epochs):
        model.train(); total=0
        for (batch_data,) in loader:
            batch_data = batch_data.to(device)
            z1 = model(augment(batch_data))
            z2 = model(augment(batch_data))
            loss = nt_xent_loss(z1, z2)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total += loss.item()
        if epoch % 200 == 0 or epoch == epochs-1:
            print(f"Epoch {epoch:4d} | Avg Loss: {total/max(1,len(loader)):.4f}")
    model.eval()
    with torch.no_grad():
        emb = model(data.to(device))
        emb = torch.nan_to_num(emb, nan=0.0, posinf=1e6, neginf=-1e6)
        return emb.cpu().numpy()



def circ_mean(angles):
    if len(angles)==0: return 0.0
    s, c = np.sin(angles).mean(), np.cos(angles).mean()
    return math.atan2(s, c)

def circ_diff(a, b):

    d = abs(a-b)
    return min(d, 2*np.pi - d)

def circ_dispersion(angles):

    if len(angles)==0: return 0.0
    s, c = np.sin(angles).mean(), np.cos(angles).mean()
    R = np.sqrt(s*s + c*c)
    return 1.0 - R

def cluster_diameter(coords):
    if len(coords)<=1: return 0.0
    center = coords.mean(axis=0)
    return 2.0*np.linalg.norm(coords-center, axis=1).max()

def safe_area(w, h, iw, ih):
    w = np.asarray(w, float); h = np.asarray(h, float)
    iw = np.asarray(iw, float); ih = np.asarray(ih, float)
    iw = np.where(iw==0, 1.0, iw); ih = np.where(ih==0, 1.0, ih)
    a = (w/iw)*(h/ih)
    a = np.where(np.isfinite(a), a, 0.0)
    return a



def initial_spatial_cluster(positions_np, min_cluster_size=5):
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
    return clusterer.fit_predict(positions_np)



def split_cluster_if_needed(
    idx_list,
    positions, embeddings, headings,   
    size_thresh, diam_thresh, heading_disp_thresh,
    min_cluster_size=4,                       
    pos_w=0, head_w=0.0,                 
    split_quantile=1e-2, 
    min_gain_ratio=1.15,
    w_pos_d_norm = 0.25   
):
  
    ids = list(idx_list)
    if len(ids) <= 2:
        return [ids]

    pts   = positions[ids]    
    embs  = embeddings[ids]    
    heads = headings[ids]      


    need_split = False
    if len(ids) >= size_thresh: need_split = True
    if cluster_diameter(pts) >= diam_thresh: need_split = True
    if circ_dispersion(heads) >= heading_disp_thresh and len(ids) >= 4:
        need_split = True
    if not need_split:
        return [ids]

  
    pos_norm  = (pts - pts.mean(axis=0)) / (np.std(pts, axis=0) + 1e-8)
    head_feat = np.stack([np.cos(heads), np.sin(heads)], axis=1)
    joint = np.concatenate([
        embs,
        pos_w  * pos_norm,
        head_w * head_feat      
    ], axis=1)

 
    joint_norm = joint / (np.linalg.norm(joint, axis=1, keepdims=True) + 1e-8)
    cos_sim = cosine_similarity(joint_norm)        
    emb_dist = 1.0 - np.clip(cos_sim, -1.0, 1.0)  

    pos_d = np.linalg.norm(pts[:,None,:] - pts[None,:,:], axis=-1)
    pos_scale = np.median(pos_d[pos_d>0]) if np.any(pos_d>0) else 1.0
    pos_d_norm = pos_d / (pos_scale + 1e-8)

 
    D = emb_dist + w_pos_d_norm * pos_d_norm
    np.fill_diagonal(D, 0.0)


    tri = D[np.triu_indices(D.shape[0], k=1)]
    if len(tri) == 0:
        return [ids]
    tau = np.quantile(tri, split_quantile)
    try:
        ac = AgglomerativeClustering(
            n_clusters=None,
            metric='precomputed',
            linkage='average',
            distance_threshold=float(tau)
        )
        labels = ac.fit_predict(D)
    except Exception:
        return [ids]

    uniq = np.unique(labels)
    if len(uniq) <= 1:
 
        if len(ids) < 4:
            return [ids]
        km = KMeans(n_clusters=2, n_init=10, random_state=0)
        kml = km.fit_predict(joint)
        if len(np.unique(kml)) == 1:
            return [ids]

        def scatter(lbls):
            c0 = joint[lbls==0]; c1 = joint[lbls==1]
            if len(c0)==0 or len(c1)==0: return np.inf
            mu0 = c0.mean(axis=0); mu1 = c1.mean(axis=0)
            w = (np.linalg.norm(c0-mu0,axis=1).mean() + np.linalg.norm(c1-mu1,axis=1).mean())
            b = np.linalg.norm(mu0-mu1)
            return w / (b+1e-8)

        orig_proxy = np.linalg.norm(joint - joint.mean(axis=0), axis=1).mean()
        new_proxy  = scatter(kml)
        if new_proxy * min_gain_ratio < orig_proxy:
            parts = [ [ids[i] for i in range(len(ids)) if kml[i]==0],
                      [ids[i] for i in range(len(ids)) if kml[i]==1] ]
            return [sorted(set(p)) for p in parts if len(p)>0]
        else:
            return [ids]

 
    parts = []
    for u in uniq:
        sub_idx = [ids[i] for i in range(len(ids)) if labels[i]==u]
        if len(sub_idx) > 0:
            parts.append(sorted(set(sub_idx)))
    return parts



def projection_separation(
    idsA, idsB, positions,
    eps=1e-6,
    norm_method="tanh",  
    scale=5.0,          
    small_cluster_floor=2
):
   
    XA = positions[idsA]
    XB = positions[idsB]
    if len(XA) == 0 or len(XB) == 0:
        return 0.0, 0.0 if norm_method else None


    muA = XA.mean(axis=0); muB = XB.mean(axis=0)
    u = muB - muA
    n = float(np.linalg.norm(u))
    if n < eps:
        return 0.0, 0.0 if norm_method else None
    u = u / n
    mid = 0.5 * (muA + muB)

    tA = ((XA - mid) @ u).astype(np.float64)
    tB = ((XB - mid) @ u).astype(np.float64)
    mA, mB = float(tA.mean()), float(tB.mean())

  
    def robust_std(t):
        if t.size >= 2:
            s = float(np.std(t, ddof=1))
            if not np.isfinite(s) or s < 0:
                s = float(np.std(t, ddof=0))
        else:
            s = 0.0
        return s

    sA = robust_std(tA); sB = robust_std(tB)

 
    if (tA.size < small_cluster_floor) or (tB.size < small_cluster_floor):
   
        t_all = np.concatenate([tA, tB])
        s_pool = float(np.std(t_all, ddof=0)) if t_all.size else 1.0
    else:
        dof = max(1, (tA.size + tB.size - 2))
        num = max(0, tA.size - 1) * (sA ** 2) + max(0, tB.size - 1) * (sB ** 2)
        s_pool = math.sqrt(num / dof) if num > 0 else 0.0

        t_all = np.concatenate([tA, tB])
        s_floor = max(eps, 0.1 * np.median(np.abs(t_all - np.median(t_all))) if t_all.size else eps)
        if s_pool < s_floor:
            s_pool = max(float(np.std(t_all, ddof=0)), s_floor)

    d_raw = abs(mA - mB) / (s_pool + eps)

    if norm_method is None:
        return d_raw, None


    if norm_method == "tanh":
        d_norm = np.tanh(d_raw / max(eps, scale))         
    elif norm_method == "mm":
        d_norm = d_raw / (d_raw + max(eps, scale))       
    elif norm_method == "logistic":
     
        k = 1.0
        d_norm = 1.0 / (1.0 + np.exp(-(d_raw - scale)/max(eps,k)))
    else:
        d_norm = None

    return d_raw, float(d_norm) if d_norm is not None else None

def area_distance_consistency(idsA, idsB, positions, areas, radius=5.0, tau=0.85):
   
    XA = positions[idsA]; XB = positions[idsB]
    AA = areas[idsA];     AB = areas[idsB]
    muA = XA.mean(axis=0); muB = XB.mean(axis=0)
    pairs = []

    for i,pa in enumerate(XA):
        dists = np.linalg.norm(XB - pa, axis=1)
        j = int(np.argmin(dists))
        if dists[j] <= radius:
            pairs.append((pa, AA[i], XB[j], AB[j]))

    for j,pb in enumerate(XB):
        dists = np.linalg.norm(XA - pb, axis=1)
        i = int(np.argmin(dists))
        if dists[i] <= radius:
            pairs.append((XA[i], AA[i], pb, AB[j]))
    if len(pairs)==0: return 0.5 
    good = 0
    for p1,a1,p2,a2 in pairs:
     
        mid = 0.5*(muA+muB)
        d1 = np.linalg.norm(p1 - mid)
        d2 = np.linalg.norm(p2 - mid)
        if d1 < d2:
            good += 1 if (a1 >= tau*a2) else 0
        elif d2 < d1:
            good += 1 if (a2 >= tau*a1) else 0
        else:
       
            good += 1 if (min(a1,a2)/max(a1,a2) >= tau) else 0
    return good / max(1,len(pairs))

def merge_with_rules(
    clusters, positions, embeddings, headings, areas,
    neighbor_radius=6.0,
    heading_opposite_deg=90.0,   
    heading_similar_deg=25.0, 
    emb_merge_thresh=0.4,         
    proj_sep_thresh=0.6,          
    area_consistency_min=0.6,    
    max_passes=5
):

    def centroid(ids):
        X = positions[ids]
        E = embeddings[ids]
        H = headings[ids]
        return X.mean(axis=0), E.mean(axis=0), circ_mean(H)

    merged = [sorted(set(ids)) for ids in clusters]
    for _ in range(max_passes):
        C = len(merged)
        if C <= 1: break
        cents = [centroid(ids) for ids in merged]
        used = [False]*C
        new_clusters = []
        changed_any = False

        centers = np.stack([c[0] for c in cents], axis=0)
        for i in range(C):
            if used[i]: continue
            base = list(merged[i])
            xi, ei, hi = cents[i]
            used[i] = True

        
            dists = np.linalg.norm(centers - xi, axis=1)
            neighbor_idx = [j for j in range(C) if (j!=i and not used[j] and dists[j] <= neighbor_radius)]

            for j in neighbor_idx:
                xj, ej, hj = cents[j]
             
                hd_deg = math.degrees(circ_diff(hi, hj))
                if hd_deg >= heading_opposite_deg:
                    continue 
                if hd_deg > heading_similar_deg:
                    continue  

           
                emb_d = np.linalg.norm(ei - ej)
                if emb_d > emb_merge_thresh:
                    continue

               

          
                lenA = len(merged[i]); lenB = len(merged[j])
                if lenA >= 2 and lenB >= 2:
                    sep_raw, sep = projection_separation(merged[i], merged[j], positions)
                    if sep > proj_sep_thresh:
                        continue

           
                base.extend(merged[j])
                used[j] = True
       
                xi = positions[base].mean(axis=0)
                ei = embeddings[base].mean(axis=0)
                hi = circ_mean(headings[base])
                changed_any = True

            new_clusters.append(sorted(set(base)))

        merged = new_clusters
        if not changed_any:
            break
    return merged

def relabel_from_clusters(clusters, N):
    labels = -1*np.ones(N, dtype=int)
    cid = 0
    for ids in clusters:
        for i in ids:
            labels[i] = cid
        cid += 1
    return labels


def run_two_clustering(
    df: pd.DataFrame,
    hidden_dim=64,
    dataset_name='COP',

    init_min_cluster_size=5,

    size_factor=2.0,           
    diam_thresh=6.0,           
    heading_disp_thresh=0.4,   
    sub_min_cluster_size=4,    

    neighbor_radius=6.0,      
    heading_opposite_deg=90.0, 
    heading_similar_deg=25.0,   
    emb_merge_thresh=0.4,
    proj_sep_thresh=0.6,
    area_consistency_min=0.6,
    max_split_merge_rounds=2
):
    classifier = df['classifier'].iloc[0]
    if dataset_name == 'COP':
        emb_cache_path = f"output/{dataset_name}/graphcl_embeddings_{classifier}.pkl"
    else:
        emb_cache_path = f"output/{dataset_name}/cl_embeddings_{classifier}.pkl"
    os.makedirs(f"output/{dataset_name}", exist_ok=True)

  
    data, pos_tensor = build_data(df)
    positions_np = pos_tensor.numpy()
    headings = df['heading'].to_numpy(dtype=float)
    areas = safe_area(df['width'].to_numpy(), df['height'].to_numpy(),
                      df['img_width'].to_numpy(), df['img_height'].to_numpy())

 
    if os.path.exists(emb_cache_path):
        print(f"[INFO] Loading cached embeddings from {emb_cache_path}")
        with open(emb_cache_path, "rb") as f:
            emb = pickle.load(f)
    else:
        emb = train_cl(data, input_dim=6, hidden_dim=hidden_dim)
        with open(emb_cache_path, "wb") as f:
            pickle.dump(emb, f)
        print(f"[INFO] Saved embeddings to {emb_cache_path}")
    embeddings_np = np.asarray(emb, dtype=np.float32)

    starttime = time.time()

    init_labels = initial_spatial_cluster(positions_np, min_cluster_size=init_min_cluster_size)
    N = len(df)
    clusters = []
    label_to_ids = {}
    for i, lb in enumerate(init_labels):
        if lb == -1:
            clusters.append([i])
        else:
            label_to_ids.setdefault(lb, []).append(i)
    clusters.extend(list(label_to_ids.values()))
    clusters = [sorted(set(ids)) for ids in clusters if len(ids)>0]

    for round_id in range(max_split_merge_rounds):
  
        sizes = [len(ids) for ids in clusters if len(ids) > 1]
        med_size = np.median(sizes) if len(sizes)>0 else 1.0
        if dataset_name == 'AAL':
            size_thresh = max(3, int(size_factor * med_size))
        if dataset_name == 'COP':
            size_thresh = max(1, int(size_factor * med_size))

        new_clusters = []
        for ids in clusters:
            if dataset_name == 'AAL':
                sub_lists = split_cluster_if_needed(
                    ids,
                    positions_np, embeddings_np, headings,
                    size_thresh=size_thresh,
                    diam_thresh=diam_thresh,
                    heading_disp_thresh=heading_disp_thresh,
                    min_cluster_size=sub_min_cluster_size,
                    pos_w=0.9, head_w=0.1,
                    split_quantile=0.8,
                    w_pos_d_norm = 0.25
                )
            if dataset_name == 'COP':
                sub_lists = split_cluster_if_needed(
                    ids,
                    positions_np, embeddings_np, headings,
                    size_thresh=size_thresh,
                    diam_thresh=diam_thresh,
                    heading_disp_thresh=heading_disp_thresh,
                    min_cluster_size=sub_min_cluster_size,
                    pos_w=0.3, head_w=0.3,
                    split_quantile=1e-2,
                    w_pos_d_norm = 0.8
                )
            new_clusters.extend(sub_lists)
        clusters = [sorted(set(x)) for x in new_clusters if len(x)>0]

    
        if dataset_name == 'AAL':
            clusters = merge_with_rules(
                clusters, positions_np, embeddings_np, headings, areas,
                neighbor_radius=neighbor_radius,
                heading_opposite_deg=heading_opposite_deg,
                heading_similar_deg=heading_similar_deg,
                emb_merge_thresh=emb_merge_thresh,
                proj_sep_thresh=proj_sep_thresh,
                area_consistency_min=area_consistency_min,
                max_passes=10
            )
        print(f"[INFO] Round {round_id+1}/{max_split_merge_rounds}: clusters={len(clusters)}")
    endtime = time.time()
    print(endtime - starttime)

    final_labels = relabel_from_clusters(clusters, N)
    out_df = df.copy()
    out_df['cluster_id'] = final_labels
    return out_df

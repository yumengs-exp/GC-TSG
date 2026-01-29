import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import haversine_distances
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import math
import os
import pandas as pd
from sklearn.preprocessing import normalize
from multiprocessing import Pool, cpu_count
# --- 距离与几何辅助函数 ---

def gps_distance(a, b):
    p1 = np.radians([[a['match_lat'], a['match_lng']]])
    p2 = np.radians([[b['match_lat'], b['match_lng']]])
    return haversine_distances(p1, p2)[0, 0] * 6371000  

def compute_heading_diff(a, b):
    diff = abs(a - b)
    return min(diff, 2 * np.pi - diff)

def mbr_similarity(a, b):
    area_a = (a['width']/a['img_width']) * (a['height']/a['img_height'])
    area_b = (b['width']/b['img_width']) * (b['height']/b['img_height'])
    ratio = max(area_a, area_b) / min(area_a, area_b)
    return ratio < 2.0



def edge_worker(args):
    i, row_i_dict, df_records, max_dist, max_angle_deg_rad = args
    edges = []
    for j in range(i + 1, len(df_records)):
        row_j = df_records[j]
        if gps_distance(row_i_dict, row_j) > max_dist:
            continue
        if compute_heading_diff(row_i_dict['heading'], row_j['heading']) > max_angle_deg_rad:
            continue
        if not mbr_similarity(row_i_dict, row_j):
            continue
        edges.append([i, j])
        edges.append([j, i])
    return edges

# COP DIST 20 ANGLE 30
def build_graph_data(df, max_dist=4, max_angle_deg=30, save_path=None):
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)

    # === Feature 构建 ===
    features = []
    for _, row in df.iterrows():
        try:
            feat = [
                getattr(row['geom'], 'x', 0.0) if pd.notnull(row['geom']) else 0.0,
                getattr(row['geom'], 'y', 0.0) if pd.notnull(row['geom']) else 0.0,
                float(np.cos(row['heading'])) if pd.notnull(row['heading']) else 1.0,
                float(np.sin(row['heading'])) if pd.notnull(row['heading']) else 0.0,
                # row['speed'] if pd.notnull(row['speed']) else 0.0,
                row['width']/row['img_width'] if pd.notnull(row['width']) and pd.notnull(row['img_width']) and row['img_width'] != 0 else 0.0,
                row['height']/row['img_height'] if pd.notnull(row['height']) and pd.notnull(row['img_height']) and row['img_height'] != 0 else 0.0,
                # row['x']/row['img_width'] if pd.notnull(row['x']) and pd.notnull(row['img_width']) and row['img_width'] != 0 else 0.0,
                # row['y']/row['img_height'] if pd.notnull(row['y']) and pd.notnull(row['img_height']) and row['img_height'] != 0 else 0.0
            ]
        except Exception as e:
            print(f"[WARN] Feature extract error: {e}")
            continue
        features.append(feat)

    if len(features) == 0:
        raise ValueError("No valid features extracted from input DataFrame.")
    
    x = torch.tensor(features, dtype=torch.float)
    if torch.isnan(x).any():
        raise ValueError("Feature tensor contains NaN values.")


    df_records = df.to_dict("records")
    max_angle_deg_rad = math.radians(max_angle_deg)
    args_list = [(i, df_records[i], df_records, max_dist, max_angle_deg_rad) for i in range(len(df))]

    edge_index = []
    with Pool(processes=cpu_count()) as pool:
        for edges in pool.imap_unordered(edge_worker, args_list):
            edge_index.extend(edges)

    if len(edge_index) == 0:
        print("[WARN] No edges found. Using self-loops.")
        edge_index = [[i, i] for i in range(len(df))]

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    if edge_index.size(1) == 0:
        raise ValueError("Edge index is empty after construction.")

    graph_data = Data(x=x, edge_index=edge_index)


    if save_path:
        torch.save(graph_data, save_path)
        print(f"[INFO] Saved built graph to: {save_path}")

    return graph_data






class GCNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        
        projection_dim = hidden_dim // 2
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x
    
    def project(self, x):
        return self.projector(x)



def graph_cl_loss(z1, z2, temperature=0.5):

    batch_size = z1.shape[0]
    z = torch.cat([z1, z2], dim=0) 
    z = F.normalize(z, dim=1) 


    sim = torch.matmul(z, z.T) 
    sim = sim / temperature


    labels = torch.arange(batch_size)
    labels = torch.cat([labels + batch_size, labels], dim=0).to(z.device)

    mask = torch.eye(2*batch_size, dtype=torch.bool).to(z.device)
    sim.masked_fill_(mask, float('-inf'))

    loss = F.cross_entropy(sim, labels)
    return loss

def perturb_graph(data, drop_edge_ratio=0.2):
    data_aug = data.clone()
    edge_index = data.edge_index


    edges = edge_index.t().tolist()  
    undirected_edges = set()
    for a, b in edges:
        if a != b:
            undirected_edges.add(tuple(sorted([a, b])))

    undirected_edges = list(undirected_edges)
    num_edges = len(undirected_edges)

    if num_edges <= 2:
        return data_aug

    num_drop = int(num_edges * drop_edge_ratio)
    if num_drop >= num_edges:
        num_drop = num_edges - 1  


    perm = torch.randperm(num_edges)
    drop_undirected_edges = set([undirected_edges[i] for i in perm[:num_drop]])


    new_edges = []
    for a, b in edges:
        edge_tuple = tuple(sorted([a, b]))
        if edge_tuple not in drop_undirected_edges:
            new_edges.append([a, b])

    data_aug.edge_index = torch.tensor(new_edges, dtype=torch.long).t().contiguous()
    return data_aug



from sklearn.preprocessing import StandardScaler

def standardize_data_x(data):

    x_np = data.x.cpu().numpy()
    scaler = StandardScaler()
    x_std = scaler.fit_transform(x_np)
    data_std = data.clone()
    data_std.x = torch.tensor(x_std, dtype=torch.float, device=data.x.device)
    return data_std, scaler


def train_graphcl(data, input_dim, hidden_dim=64, epochs=2000, lr=1e-1):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Using device: {device}")


    data, scaler = standardize_data_x(data)
    data = data.to(device)  

 
    model = GCNEncoder(input_dim, hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()

    for epoch in range(epochs):
  
        data1 = perturb_graph(data).to(device)
        data2 = perturb_graph(data).to(device)

        z1 = model(data1.x, data1.edge_index)
        z2 = model(data2.x, data2.edge_index)

        z1 = model.project(z1)
        z2 = model.project(z2)

        if torch.isnan(z1).any() or torch.isnan(z2).any():
            print(f"[WARN] NaN detected in embeddings at epoch {epoch}, skipping update.")
            continue

        loss = graph_cl_loss(z1, z2)

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"[WARN] Loss is NaN/Inf at epoch {epoch}, skipping.")
            continue

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        emb = model(data.x, data.edge_index)
        emb = torch.nan_to_num(emb, nan=0.0, posinf=1e6, neginf=-1e6)
        return emb.cpu().numpy() 


def cluster_embeddings(emb, eps=0.5, min_samples=5):

    emb_normalized = normalize(emb, norm='l2')


    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
    labels = clustering.fit_predict(emb_normalized)
    return labels

def cluster_point(df, eps=0.5, min_samples=5):


    def _point_to_array(geom: Point):
        return np.array([geom.x, geom.y])
    generator = map(_point_to_array, df['geom'])
    generator = map(lambda coords, heading: np.array([coords[0], coords[1], heading]), generator, df['heading'])
    clustering = DBSCAN(eps=eps, min_samples=min_samples)
    labels = clustering.fit_predict(np.array(list(generator)))
    return labels





from shapely.geometry import Point
import shapely.wkt
import os
import pandas as pd
import os
import pickle
import shapely.wkt
from shapely.geometry import Point
import pandas as pd

def run_graphcl_clustering(df: pd.DataFrame, hidden_dim=64, cluster_eps=0.2, min_samples=1, dataset_name='COP'):
    classifier = df['classifier'].iloc[0]
    emb_cache_path = f"output/{dataset_name}/graphcl_embeddings_{classifier}.pkl"
    graph_cache_path = f"output/{dataset_name}/graph_data_{classifier}.pt"
    os.makedirs(f"output/{dataset_name}", exist_ok=True)

    if os.path.exists(emb_cache_path):
        print(f"[INFO] Loading cached embeddings from {emb_cache_path}")
        with open(emb_cache_path, "rb") as f:
            emb = pickle.load(f)

    else:
        if os.path.exists(graph_cache_path):
            print(f"[INFO] Loading cached graph from {graph_cache_path}")
            data = torch.load(graph_cache_path, weights_only=False)

        else:
            print(f"[INFO] Building graph for classifier '{classifier}'...")
            data = build_graph_data(df, save_path=graph_cache_path)

        emb = train_graphcl(data, input_dim=data.num_features, hidden_dim=hidden_dim)

        with open(emb_cache_path, "wb") as f:
            pickle.dump(emb, f)
        print(f"[INFO] Saved embeddings to {emb_cache_path}")

    labels = cluster_embeddings(emb, eps=cluster_eps, min_samples=min_samples)

    df = df.copy()
    df['cluster_id'] = labels

    return df



def run_dbscan_clustering(df: pd.DataFrame, hidden_dim=64, cluster_eps=0.2, min_samples=1, dataset_name='COP'):
    labels = cluster_point(df, eps=cluster_eps, min_samples=min_samples)

    df = df.copy()
    df['cluster_id'] = labels

    return df




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




def gps_distance(a, b):
    p1 = np.radians([[a['match_lat'], a['match_lng']]])
    p2 = np.radians([[b['match_lat'], b['match_lng']]])
    return haversine_distances(p1, p2)[0, 0] * 6371000 

def compute_heading_diff(a, b):
    diff = abs(a - b)
    return min(diff, 2 * np.pi - diff)


def mbr_similarity(a, b):
    area_a = (a['width'] / a['img_width']) * (a['height'] / a['img_height'])
    area_b = (b['width'] / b['img_width']) * (b['height'] / b['img_height'])
    ratio = max(area_a, area_b) / min(area_a, area_b)
    return ratio < 2.0



def build_data(df, save_path=None):
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    import pandas as pd
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

    x = torch.tensor(features, dtype=torch.float)  # shape: [N, 6]

    if torch.isnan(x).any():
        raise ValueError("Feature tensor contains NaN values.")


    x_normed = x / (x.sum(dim=1, keepdim=True) + 1e-8)


    

    return x_normed, torch.tensor(positions, dtype=torch.float)  





def augment(x, noise_std=0.01):

    noise = torch.randn_like(x) * noise_std
    return x + noise


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(MLP, self).__init__()
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
    z = torch.cat([z1, z2], dim=0)  # [2N, D]
    sim_matrix = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)  # [2N, 2N]

   
    labels = torch.arange(N).to(z.device)
    labels = torch.cat([labels, labels], dim=0)

    logits = sim_matrix / temperature
    loss = F.cross_entropy(logits, labels)
    return loss


from torch.utils.data import DataLoader, TensorDataset

def train_cl(data, input_dim, hidden_dim=16, epochs=3000, lr=1e-1, batch_size=512):
    """
    data: torch.Tensor, shape [N, input_dim]
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MLP(input_dim, hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    dataset = TensorDataset(data)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch in loader:
            batch_data = batch[0].to(device)

        
            x1 = augment(batch_data)
            x2 = augment(batch_data)

      
            z1 = model(x1)
            z2 = model(x2)

        
            loss = nt_xent_loss(z1, z2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if epoch % 200 == 0 or epoch == epochs - 1:
            avg_loss = total_loss / len(loader)
            print(f"Epoch {epoch:4d} | Avg Loss: {avg_loss:.4f}")

 
    model.eval()
    data = data.to(device)
    with torch.no_grad():
        emb = model(data)
        emb = torch.nan_to_num(emb, nan=0.0, posinf=1e6, neginf=-1e6)
        return emb.cpu().numpy()




import hdbscan


# aal: 
def cluster_embeddings(embeddings, positions=None, min_cluster_size=5, alpha=0.4):



    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu().numpy()
    if positions is not None and isinstance(positions, torch.Tensor):
        positions = positions.detach().cpu().numpy()

    if positions is not None:
        
        epsilon = 1e-8

      
        joint_features = np.concatenate([(1-alpha) * embeddings, alpha * positions], axis=1)
    else:
        joint_features = embeddings  

    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
    labels = clusterer.fit_predict(joint_features)

    return labels



def cluster_point(df, eps=0.5, min_samples=5):
    def _point_to_array(geom: Point):
        return np.array([geom.x, geom.y])

    generator = map(_point_to_array, df['geom'])
    generator = map(lambda coords, heading: np.array([coords[0], coords[1], heading]), generator, df['heading'])
    clustering = DBSCAN(eps=eps, min_samples=min_samples)
    labels = clustering.fit_predict(np.array(list(generator)))
    return labels

def hcluster_point(df, min_samples=5):
    def _point_to_array(geom: Point):
        return np.array([geom.x, geom.y])

    generator = map(_point_to_array, df['geom'])
    generator = map(lambda coords, heading: np.array([coords[0], coords[1], heading]), generator, df['heading'])
    clustering = hdbscan.HDBSCAN(min_cluster_size=min_samples)
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


def run_cl_clustering(df: pd.DataFrame, hidden_dim=64,  dataset_name='COP'):
    classifier = df['classifier'].iloc[0]
    emb_cache_path = f"output/{dataset_name}/cl_embeddings_{classifier}.pkl"
    os.makedirs(f"output/{dataset_name}", exist_ok=True)
    data, pos_tensor = build_data(df)
    if os.path.exists(emb_cache_path):
        print(f"[INFO] Loading cached embeddings from {emb_cache_path}")
        with open(emb_cache_path, "rb") as f:
            emb = pickle.load(f)

    else:


        emb = train_cl(data, input_dim=6, hidden_dim=hidden_dim)

        with open(emb_cache_path, "wb") as f:
            pickle.dump(emb, f)
        print(f"[INFO] Saved embeddings to {emb_cache_path}")

    labels = cluster_embeddings(emb,pos_tensor)

    df = df.copy()
    df['cluster_id'] = labels

    return df


def run_dbscan_clustering(df: pd.DataFrame, hidden_dim=64, cluster_eps=0.2, min_samples=1, dataset_name='COP'):
    labels = cluster_point(df, eps=cluster_eps, min_samples=min_samples)

    df = df.copy()
    df['cluster_id'] = labels

    return df

def run_hdbscan_clustering(df: pd.DataFrame, min_samples=1):
    labels = hcluster_point(df, min_samples=min_samples)

    df = df.copy()
    df['cluster_id'] = labels

    return df


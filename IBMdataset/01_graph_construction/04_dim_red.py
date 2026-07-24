import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import hdbscan

centroids = np.load('policy_centroids.npy')
policy_df = pd.read_csv('policies.csv')

# --- Step 1: Reduce dimensions ---
pca = PCA(n_components=20, random_state=42)
reduced = pca.fit_transform(centroids)
print(f"Variance explained: {pca.explained_variance_ratio_.sum():.3f}")

# --- Step 2: KMeans on reduced space ---
print("\nKMeans on PCA-reduced centroids:")
for k in range(5, 16):
    km = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels = km.fit_predict(reduced)
    score = silhouette_score(reduced, labels)
    print(f"k={k:2d} | silhouette={score:.4f}")

# --- Step 3: HDBSCAN on reduced space ---
print("\nHDBSCAN on PCA-reduced centroids:")
for min_cluster_size in [3, 4, 5, 6]:
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric='euclidean'
    )
    labels = clusterer.fit_predict(reduced)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
    if n_clusters > 1:
        # Only compute silhouette on non-noise points
        mask = labels != -1
        if mask.sum() > n_clusters:
            score = silhouette_score(reduced[mask], labels[mask])
        else:
            score = float('nan')
    else:
        score = float('nan')
        
    print(f"min_cluster_size={min_cluster_size} | "
          f"clusters={n_clusters} | "
          f"noise={n_noise} | "
          f"silhouette={score:.4f}" if not np.isnan(score) 
          else f"min_cluster_size={min_cluster_size} | "
               f"clusters={n_clusters} | noise={n_noise} | silhouette=N/A")
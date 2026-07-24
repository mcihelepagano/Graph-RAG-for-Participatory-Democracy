import numpy as np
import pandas as pd

# --- 1. Load data ---
df = pd.read_csv('arguments_clean.csv')
embeddings = np.load('argument_embeddings.npy')

print(f"Arguments: {len(df)}")
print(f"Embedding matrix: {embeddings.shape}")

# --- 2. Compute one centroid per policy ---
policies = df['policy'].unique()
centroids = []

for policy in policies:
    # Get indices of all arguments belonging to this policy
    indices = df[df['policy'] == policy].index.tolist()
    policy_embeddings = embeddings[indices]
    
    # Average all argument embeddings → one centroid vector
    centroid = policy_embeddings.mean(axis=0)
    
    # Re-normalize (averaging breaks unit norm)
    centroid = centroid / np.linalg.norm(centroid)
    
    centroids.append({
        'policy': policy,
        'n_arguments': len(indices)
    })

centroid_matrix = np.array([
    embeddings[df[df['policy'] == p].index].mean(axis=0) 
    for p in policies
])

# Re-normalize all centroids
norms = np.linalg.norm(centroid_matrix, axis=1, keepdims=True)
centroid_matrix = centroid_matrix / norms

# --- 3. Save ---
policy_df = pd.DataFrame(centroids)
policy_df.to_csv('policies.csv', index=False)
np.save('policy_centroids.npy', centroid_matrix)

# --- 4. Sanity check ---
print(f"\nPolicy centroid matrix shape: {centroid_matrix.shape}")  # expect (71, 768)
print(f"Sample norms (should all be ~1.0): {np.linalg.norm(centroid_matrix[:5], axis=1)}")
print(f"\nArguments per policy:")
print(policy_df['n_arguments'].describe())
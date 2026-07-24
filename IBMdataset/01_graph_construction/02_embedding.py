from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np

# --- 1. Load data ---
df = pd.read_csv('arguments_clean.csv')
print(f"Loaded {len(df)} arguments across {df['policy'].nunique()} policies")

# --- 2. Load model ---
model = SentenceTransformer("all-mpnet-base-v2")

# --- 3. Embed all arguments ---
print("Embedding arguments...")
embeddings = model.encode(
    df['argument'].tolist(),
    batch_size=256,
    show_progress_bar=True,
    normalize_embeddings=True
)

print(f"Embedding matrix shape: {embeddings.shape}")  # expect (30497, 768)

# --- 4. Save embeddings ---
np.save('argument_embeddings.npy', embeddings)
print("Saved argument_embeddings.npy")

# --- 5. Add index column to CSV ---
df['embedding_idx'] = range(len(df))
df.to_csv('arguments_clean.csv', index=False)
print("Updated arguments_clean.csv with embedding_idx")

# --- 6. Sanity check ---
loaded = np.load('argument_embeddings.npy')
assert loaded.shape == embeddings.shape
assert abs(np.linalg.norm(loaded[0]) - 1.0) < 1e-5  # verify normalization
print("Sanity check passed. All embeddings unit-normalized.")
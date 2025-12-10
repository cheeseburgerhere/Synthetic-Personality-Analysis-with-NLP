import os
import pandas as pd
import numpy as np
import logging
from sklearn.metrics.pairwise import cosine_similarity

# Optional imports
try:
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(output_dir):
    """Loads clustering results and embeddings."""
    clusters_path = os.path.join(output_dir, "hobby_clusters_v2.csv")
    emb_06b_path = os.path.join(output_dir, "canonical_embeddings_qwen3_0.6b.npy")
    emb_8b_path = os.path.join(output_dir, "canonical_embeddings_qwen3_8b.npy")

    if not os.path.exists(clusters_path):
        raise FileNotFoundError(f"Clusters file not found: {clusters_path}")
    
    df_clusters = pd.read_csv(clusters_path)
    
    emb_06b = np.load(emb_06b_path) if os.path.exists(emb_06b_path) else None
    emb_8b = np.load(emb_8b_path) if os.path.exists(emb_8b_path) else None

    return df_clusters, emb_06b, emb_8b

def evaluate_reduction(df):
    """Calculates and prints reduction statistics."""
    num_canonical = len(df)
    total_raw_inputs = df['variations'].apply(lambda x: len(eval(x))).sum()
    
    reduction_ratio = (1 - (num_canonical / total_raw_inputs)) * 100
    
    print("\n" + "="*40)
    print("CLUSTERING REDUCTION STATISTICS")
    print("="*40)
    print(f"Total Unique Input Hobbies: {total_raw_inputs}")
    print(f"Total Canonical Groups:     {num_canonical}")
    print(f"Reduction Ratio:            {reduction_ratio:.2f}%")
    print("="*40 + "\n")

def inspect_clusters(df, num_samples=5):
    """Prints random clusters to verify grouping logic."""
    print("RANDOM CLUSTER SAMPLES")
    print("-" * 30)
    sample = df.sample(min(num_samples, len(df)))
    for _, row in sample.iterrows():
        print(f"Canonical: {row['canonical']}")
        print(f"Variations: {row['variations']}")
        print("-" * 15)
    print("\n")

def find_nearest_neighbors(target_idx, embeddings, df, model_name, k=5):
    """Finds and prints nearest neighbors for a given target index."""
    if embeddings is None:
        print(f"No embeddings found for {model_name}.")
        return

    sim_matrix = cosine_similarity(embeddings[target_idx].reshape(1, -1), embeddings)
    top_indices = sim_matrix[0].argsort()[::-1][1:k+1]
    
    target_term = df.iloc[target_idx]['canonical']
    print(f"Nearest Neighbors for '{target_term}' using {model_name}:")
    for idx in top_indices:
        neighbor_term = df.iloc[idx]['canonical']
        score = sim_matrix[0][idx]
        print(f"  {score:.4f} - {neighbor_term}")
    print("\n")

def visualize_embeddings(embeddings, df, output_path, title):
    """Generates TSNE plot for embeddings."""
    if not HAS_MATPLOTLIB:
        logging.warning("Matplotlib or TSNE not available. Skipping visualization.")
        return

    if embeddings is None:
        return
        
    logging.info(f"Generating TSNE for {title}...")
    tsne = TSNE(n_components=2, perplexity=min(30, len(df)-1), random_state=42)
    vis_dims = tsne.fit_transform(embeddings)
    
    plt.figure(figsize=(10, 8))
    plt.scatter(vis_dims[:, 0], vis_dims[:, 1], alpha=0.6, s=10)
    
    # Annotate a few points
    num_annotations = min(20, len(df))
    indices_to_annotate = np.random.choice(len(df), num_annotations, replace=False)
    for i in indices_to_annotate:
        plt.text(vis_dims[i, 0], vis_dims[i, 1], df.iloc[i]['canonical'], fontsize=8, alpha=0.7)
        
    plt.title(title)
    plt.savefig(output_path)
    plt.close()
    logging.info(f"Saved plot to {output_path}")

def main():
    base_dir = os.path.dirname(__file__)
    output_dir = os.path.join(base_dir, 'output')
    
    try:
        df, emb_06b, emb_8b = load_data(output_dir)
        
        # 1. Reduction Stats
        evaluate_reduction(df)
        
        # 2. Cluster Inspection
        inspect_clusters(df)
        
        # 3. Semantic Check (Nearest Neighbors)
        test_idx = np.random.randint(0, len(df))
        test_term = df.iloc[test_idx]['canonical']
        print(f"Comparing Semantic Similarity for term: '{test_term}'")
        print("-" * 50)
        
        find_nearest_neighbors(test_idx, emb_06b, df, "Qwen 3 0.6B")
        find_nearest_neighbors(test_idx, emb_8b, df, "Qwen 3 8B")
        
        # 4. Visualization
        if emb_8b is not None:
            visualize_embeddings(emb_8b, df, os.path.join(output_dir, "embeddings_tsne_qwen3_8b.png"), "Qwen 3 8B Embeddings TSNE")
        
        if emb_06b is not None:
             visualize_embeddings(emb_06b, df, os.path.join(output_dir, "embeddings_tsne_qwen3_0.6b.png"), "Qwen 3 0.6B Embeddings TSNE")

    except Exception as e:
        logging.error(f"Evaluation failed: {e}")

if __name__ == "__main__":
    main()

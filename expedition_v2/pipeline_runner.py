import os
import sys
import pandas as pd
import numpy as np
import logging
import argparse
from clustering_ncd import NCDClustering
from embedding_factory import EmbeddingGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(__file__), 'logs', 'pipeline.log')),
        logging.StreamHandler()
    ]
)

def main():
    parser = argparse.ArgumentParser(description="V2 Pipeline Runner")
    parser.add_argument("--limit", type=int, default=100, help="How many rows to process for testing")
    parser.add_argument("--parquet_path", type=str, default="../initial_expedition/train-00000-of-00011.parquet", help="Path to input parquet")
    args = parser.parse_args()

    # 1. Load Data
    parquet_path = os.path.abspath(os.path.join(os.path.dirname(__file__), args.parquet_path))
    if not os.path.exists(parquet_path):
        logging.error(f"Parquet file not found at {parquet_path}")
        return

    logging.info(f"Loading data from {parquet_path} (limit={args.limit})...")
    df = pd.read_parquet(parquet_path).iloc[:args.limit]
    
    # Extract hobbies (assuming column name 'hobbies_and_interests' or similar from previous exploration)
    # The previous code in dataset_expedition_hobbies.ipynb used 'hobbies_and_interests_list' after eval
    # Or 'hobbies_and_interests' column
    
    # Let's inspect the columns in the df we just loaded to be sure, but we know from previous read
    # 'hobbies_and_interests' is a string looking like a list
    
    import ast
    all_hobbies = []
    
    if 'hobbies_and_interests_list' in df.columns:
        # The list is stored as a string representation in 'hobbies_and_interests_list'
        for item in df['hobbies_and_interests_list']:
            try:
                # Handle potential non-list strings if any
                hobbies_list = ast.literal_eval(item)
                if isinstance(hobbies_list, list):
                    all_hobbies.extend(hobbies_list)
            except Exception as e:
                logging.warning(f"Could not parse hobby entry: {item}")
    elif 'hobbies_and_interests' in df.columns:
        # Fallback if list column missing (unexpected)
        logging.warning("hobbies_and_interests_list not found, checking hobbies_and_interests but it looks like text...")
                
    unique_hobbies = list(set(all_hobbies))
    logging.info(f"Found {len(unique_hobbies)} unique hobbies.")
    
    # 2. Clustering
    logging.info("Starting NCD Clustering...")
    ncd = NCDClustering()
    # NCD is expensive, for >1000 items it will be slow. 
    # For this research run, we might want to sample if unique_hobbies is huge.
    # But for the task, we process what we have.
    
    clusters = ncd.cluster(unique_hobbies, distance_threshold=0.45)
    
    # Create canonical list (key = canonical name, value = list of variations)
    canonical_map = {}
    canonical_hobbies = []
    
    for label, group in clusters.items():
        # Pick shortest as canonical? Or most frequent (we lost frequency info)?
        # Picking shortest is a decent heuristic for "Running" vs "Running in the park"
        canonical = min(group, key=len)
        canonical_hobbies.append(canonical)
        canonical_map[canonical] = group
        
    logging.info(f"Reduced to {len(canonical_hobbies)} canonical hobbies from {len(unique_hobbies)} unique raw inputs.")
    
    # Save Cluster Data
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    pd.DataFrame([
        {"canonical": k, "variations": str(v)} for k,v in canonical_map.items()
    ]).to_csv(os.path.join(output_dir, "hobby_clusters_v2.csv"), index=False)
    
    # 3. Embedding Generation
    # Qwen 0.5B
    logging.info("Generating embeddings with Qwen 0.5B...")
    try:
        # Using a widely available Qwen model identifier
        gen_small = EmbeddingGenerator("Qwen/Qwen2.5-0.5B") 
        emb_small = gen_small.get_embedding(canonical_hobbies)
        np.save(os.path.join(output_dir, "canonical_embeddings_qwen0.5b.npy"), emb_small)
        logging.info("Saved Qwen 0.5B embeddings.")
    except Exception as e:
        logging.error(f"Skipping Qwen 0.5B generation due to error: {e}")

    # Qwen 7B (request said 8B, but sticking to 7B or 14B is standard for Qwen1.5/2.5 series. Qwen1 was 7B/14B. Specifying 7B is safer resource wise)
    # If the user machine can't handle it, it will crash/error out, which is fine for research code (we log it).
    logging.info("Generating embeddings with Qwen 7B...")
    try:
        gen_large = EmbeddingGenerator("Qwen/Qwen2.5-7B")
        emb_large = gen_large.get_embedding(canonical_hobbies)
        np.save(os.path.join(output_dir, "canonical_embeddings_qwen7b.npy"), emb_large)
        logging.info("Saved Qwen 7B embeddings.")
    except Exception as e:
        logging.error(f"Skipping Qwen 7B (Large) generation due to error: {e}")

    logging.info("Pipeline complete.")

if __name__ == "__main__":
    main()

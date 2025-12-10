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
    parser.add_argument("--input_path", type=str, default="../all_hobbies.json", help="Path to input data (parquet or json)")
    args = parser.parse_args()

    # 1. Load Data
    input_path = os.path.abspath(os.path.join(os.path.dirname(__file__), args.input_path))
    if not os.path.exists(input_path):
        logging.error(f"Input file not found at {input_path}")
        return

    logging.info(f"Loading data from {input_path} (limit={args.limit})...")
    
    unique_hobbies = []
    
    if input_path.endswith('.json'):
        import json
        try:
            with open(input_path, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    # We assume it is a list of strings as seen in inspection
                    unique_hobbies = list(set(data))
                    # Apply limit if needed (though for a list of strings, limit usually applies to *items* processing)
                    # For consistency with "limit", we can slice the list.
                    if args.limit > 0:
                        unique_hobbies = unique_hobbies[:args.limit]
                else:
                    logging.error("JSON content is not a list.")
                    return
        except Exception as e:
            logging.error(f"Failed to load JSON: {e}")
            return
            
    elif input_path.endswith('.parquet'):
        df = pd.read_parquet(input_path).iloc[:args.limit]
        
        # Extract hobbies logic (legacy support)
        import ast
        all_hobbies = []
        if 'hobbies_and_interests_list' in df.columns:
            for item in df['hobbies_and_interests_list']:
                try:
                    hobbies_list = ast.literal_eval(item)
                    if isinstance(hobbies_list, list):
                        all_hobbies.extend(hobbies_list)
                except Exception as e:
                    logging.warning(f"Could not parse hobby entry: {item}")
        elif 'hobbies_and_interests' in df.columns:
            logging.warning("hobbies_and_interests_list not found, using text column if needed (not implemented)")
            
        unique_hobbies = list(set(all_hobbies))
    else:
        logging.error("Unsupported file extension. Use .json or .parquet")
        return

    logging.info(f"Processing {len(unique_hobbies)} unique hobbies.")
    
    # 2. Clustering
    logging.info("Starting NCD Clustering...")
    ncd = NCDClustering()
    
    clusters = ncd.cluster(unique_hobbies, distance_threshold=0.45)
    
    # Create canonical list
    canonical_map = {}
    canonical_hobbies = []
    
    for label, group in clusters.items():
        canonical = min(group, key=len)
        canonical_hobbies.append(canonical)
        canonical_map[canonical] = group
        
    logging.info(f"Reduced to {len(canonical_hobbies)} canonical hobbies from {len(unique_hobbies)} unique input items.")
    
    # Save Cluster Data
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(output_dir, exist_ok=True)
    pd.DataFrame([
        {"canonical": k, "variations": str(v)} for k,v in canonical_map.items()
    ]).to_csv(os.path.join(output_dir, "hobby_clusters_v2.csv"), index=False)
    
    # 3. Embedding Generation
    
    # Qwen 3 Small (0.5B -> 0.6B)
    # The user specified Qwen/Qwen3-Embedding-0.6B
    logging.info("Generating embeddings with Qwen 0.6B...")
    try:
        gen_small = EmbeddingGenerator("Qwen/Qwen3-Embedding-0.6B") 
        emb_small = gen_small.get_embedding(canonical_hobbies)
        np.save(os.path.join(output_dir, "canonical_embeddings_qwen3_0.6b.npy"), emb_small)
        logging.info("Saved Qwen 0.6B embeddings.")
    except Exception as e:
        logging.error(f"Skipping Qwen 0.6B generation due to error: {e}")

    # Qwen 3 Large (7B -> 8B)
    # The user specified Qwen/Qwen3-Embedding-8B
    logging.info("Generating embeddings with Qwen 8B...")
    try:
        gen_large = EmbeddingGenerator("Qwen/Qwen3-Embedding-8B")
        emb_large = gen_large.get_embedding(canonical_hobbies)
        np.save(os.path.join(output_dir, "canonical_embeddings_qwen3_8b.npy"), emb_large)
        logging.info("Saved Qwen 8B embeddings.")
    except Exception as e:
        logging.error(f"Skipping Qwen 8B generation due to error: {e}")

    logging.info("Pipeline complete.")

if __name__ == "__main__":
    main()

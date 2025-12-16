import numpy as np
import pandas as pd
import pickle
import os
import sys
import tempfile
import re

# Add the parent directory to sys.path to allow imports from sibling directories
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from prediction.prediction_MatrixKNN import PredictionMatrixKNN

def load_persona_embeddings(embeds_dir):
    """
    Loads and concatenates persona embeddings from pickle files.
    Assumes filenames are in the format: persona_embeddings_Qwen.Qwen3-Embedding-8B_START-END.pkl
    """
    files = [f for f in os.listdir(embeds_dir) if f.endswith('.pkl') and "persona_embeddings" in f]
    
    # Sort files based on the starting index in the filename
    def transform_key(filename):
        match = re.search(r'_(\d+)-(\d+)\.pkl', filename)
        if match:
            return int(match.group(1))
        return 0
        
    files.sort(key=transform_key)
    
    print(f"Found {len(files)} embedding files: {files}")
    
    all_embeddings = []
    for f in files:
        path = os.path.join(embeds_dir, f)
        print(f"Loading {path}...")
        with open(path, 'rb') as pkl_file:
            data = pickle.load(pkl_file)
            if isinstance(data, list):
                all_embeddings.append(np.array(data))
            elif isinstance(data, np.ndarray):
                all_embeddings.append(data)
            else:
                 print(f"Warning: Unknown data type in {f}: {type(data)}")

    if not all_embeddings:
        raise ValueError("No embeddings loaded.")
        
    X_raw = np.concatenate(all_embeddings, axis=0)
    print(f"Total loaded persona embeddings (X): {X_raw.shape}")
    return X_raw

def main():
    # Paths
    embeds_dir = os.path.join(parent_dir, "embeds", "v0.1_8B")
    hobby_csv_path = os.path.join(parent_dir, "expedition_v2", "output", "hobby_clusters_v2.csv")
    hobby_emb_path = os.path.join(parent_dir, "expedition_v2", "output", "canonical_embeddings_qwen3_8b.npy")
    output_path = os.path.join(current_dir, "projection_dataset.npz")

    # 1. Load Persona Vectors (X)
    X_raw = load_persona_embeddings(embeds_dir)
    
    # 2. Prepare Hobby Data (Temporary CSV correction)
    print("Preparing hobby data...")
    df = pd.read_csv(hobby_csv_path)
    if "canonical" in df.columns and "canonical_hobby" not in df.columns:
        print("Renaming 'canonical' to 'canonical_hobby' for compatibility...")
        df.rename(columns={"canonical": "canonical_hobby"}, inplace=True)
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as tmp_csv:
        df.to_csv(tmp_csv.name, index=False)
        tmp_csv_path = tmp_csv.name
        
    try:
        # 3. Initialize MatrixKNN
        print("Initializing PredictionMatrixKNN...")
        predictor = PredictionMatrixKNN(hobby_emb_path, tmp_csv_path)
        
        # 4. Project (Get Indices) - k=1 for 1-to-1 mapping
        print("Projecting personas to hobbies (k=1)...")
        # predictor.predict returns list of strings, but we need indices/vectors.
        # We access the internal knn object directly as planned.
        # search_cosine returns (scores, indices)
        scores, indices = predictor.knn.search_cosine(X_raw, k=1)
        
        # indices shape is expected to be (N, 1) since k=1
        print(f"Projection indices shape: {indices.shape}")
        
        # 5. Construct Dataset (X, y)
        X = X_raw # (N, D)
        
        # Flatten indices to shape (N,)
        indices_flat = indices.flatten()
        
        # Retrieve hobby vectors (y)
        # predictor.hobbies_embeddings is (M, D)
        y = predictor.hobbies_embeddings[indices_flat]
        
        print(f"Final X shape: {X.shape}")
        print(f"Final y shape: {y.shape}")
        
        if X.shape[0] != y.shape[0]:
             raise ValueError(f"Mismatch in X and y dimensions: {X.shape[0]} vs {y.shape[0]}")
             
        # 6. Save
        print(f"Saving dataset to {output_path}...")
        np.savez(output_path, X=X, y=y)
        print("Done.")

    finally:
        # Cleanup temp file
        if os.path.exists(tmp_csv_path):
            try:
                os.remove(tmp_csv_path)
            except:
                pass

if __name__ == "__main__":
    main()

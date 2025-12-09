import numpy as np
import pandas as pd
import sys
import os

# Ensure we can import from the same directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from similarity_utils import MatrixKNN
from base_predictor import BasePredictor

#This method looks great but after some tests we have found that it leans too much into general work experience 
#because of this, users can feel unseen because it lacks nuance

class PredictionMatrixKNN(BasePredictor):
    def __init__(self, embeddings_path, csv_path):
        """
        Initialize the MatrixKNN predictor.
        
        Args:
            embeddings_path (str): Path to the .npy file containing hobby embeddings.
            csv_path (str): Path to the .csv file containing hobby descriptions.
        """
        if not os.path.exists(embeddings_path):
            raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        print(f"Loading embeddings from {embeddings_path}...")
        self.hobbies_embeddings = np.load(embeddings_path, allow_pickle=True)
        
        print(f"Loading hobby list from {csv_path}...")
        self.hobby_df = pd.read_csv(csv_path)
        
        if "canonical_hobby" not in self.hobby_df.columns:
             raise ValueError("CSV must contain 'canonical_hobby' column")
             
        # Convert to numpy array for efficient indexing
        self.hobby_list = self.hobby_df["canonical_hobby"].values
        
        print("Initializing MatrixKNN...")
        self.knn = MatrixKNN(self.hobbies_embeddings)
        print("Initialization complete.")

    def predict(self, vectors, k=5, **kwargs):
        """
        Predict hobbies for given persona vectors.
        
        Args:
            vectors (np.array): Query vectors of shape (N, D) or (D,).
            k (int): Number of nearest neighbors to return.
            **kwargs: Additional arguments (ignored in this implementation).
            
        Returns:
            list of list of str: Predicted hobbies for each query vector.
        """
        # Search returns (similarities, indices)
        # Note: search_cosine expects shape handling inside it basically, but returns (N, k)
        scores, indices = self.knn.search_cosine(vectors, k=k)
        
        results = []
        for row_idx in indices:
            # row_idx contains the indices of the neighbors for one query
            predicted_hobbies = self.hobby_list[row_idx]
            results.append(predicted_hobbies.tolist())
            
        return results

if __name__ == "__main__":
    # Simple CLI test
    pass

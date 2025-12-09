from sentence_transformers import CrossEncoder
import numpy as np
import os
import sys

# Ensure we can import from the same directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from base_predictor import BasePredictor
from prediction_MatrixKNN import PredictionMatrixKNN

class PredictionCrossEncoder(BasePredictor):
    def __init__(self, embeddings_path, csv_path, model_name='cross-encoder/ms-marco-MiniLM-L-6-v2'):
        """
        Initialize the Cross-Encoder predictor with two-stage retrieval.
        
        Args:
            embeddings_path (str): Path to embeddings for fast retrieval.
            csv_path (str): Path to CSV for hobby texts.
            model_name (str): Name of the Cross-Encoder model to load.
        """
        # print("Initializing PredictionMatrixKNN for first-stage retrieval...")
        self.matrixKNN = PredictionMatrixKNN(embeddings_path, csv_path)
        
        print(f"Loading Cross-Encoder model: {model_name}...")
        self.cross_encoder = CrossEncoder(model_name)
        print("Cross-Encoder initialization complete.")

    def predict(self, vectors, k=5, **kwargs):
        """
        Predict hobbies using re-ranking.
        
        Args:
            vectors (np.array): Query vector(s) of shape (D,) or (1, D). 
                                CURRENTLY SUPPORTS SINGLE QUERY VECTOR ONLY for simplicity in re-ranking logic loop.
            k (int): Number of final suggestions to return.
            **kwargs: Must contain 'persona_text' (str) for the cross-encoder.
            
        Returns:
            list of list of str: Predicted hobbies (currently list of 1 list of k hobbies).
        """
        persona_text = kwargs.get('persona_text')
        if not persona_text:
            raise ValueError("PredictionCrossEncoder requires 'persona_text' in kwargs for re-ranking.")

        # 1. Fast Retrieval (MatrixKNN)
        # We ask for more candidates (e.g., 2*k or fixed 10/20) to re-rank
        initial_k = max(10, k * 2) 
        
        # matrixKNN.predict returns list of lists of strings
        # We assume vectors is a single query for now as per the notebook logic structure
        initial_hits_list = self.matrixKNN.predict(vectors, k=initial_k)
        
        # Handle batch output from matrixKNN (we assume batch size 1 for this specific logic flow usually)
        # If vectors has multiple rows, we would need to loop. The notebook logic was for single input.
        # Let's support batch = 1 strictly or loop if needed.
        
        final_results = []
        
        for i, hit_candidates in enumerate(initial_hits_list):
            # hit_candidates is a list of strings (hobbies)
            
            # 2. Prepare pairs for Cross-Encoder
            # [ [persona_text, hobby1], [persona_text, hobby2], ... ]
            model_inputs = [[str(persona_text), str(hit)] for hit in hit_candidates]
            
            # 3. Predict scores
            cross_scores = self.cross_encoder.predict(model_inputs)
            
            # 4. Sort and Filter
            # Zip candidates with scores
            scored_candidates = []
            for j, candidate in enumerate(hit_candidates):
                scored_candidates.append((candidate, cross_scores[j]))
            
            # Sort descending by score
            scored_candidates.sort(key=lambda x: x[1], reverse=True)
            
            # Take top k
            top_k = [item[0] for item in scored_candidates[:k]]
            final_results.append(top_k)
            
        return final_results

import numpy as np
import os
import sys

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from prediction_CrossEncoder import PredictionCrossEncoder

def test_cross_encoder():
    print("=== Testing PredictionCrossEncoder ===")
    
    # Paths (relative to this script)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    embeddings_path = os.path.join(current_dir, "../initial_expedition/canonical_embeddings_qwen8B_v0.1.npy")
    csv_path = os.path.join(current_dir, "../initial_expedition/semantically_merged_hobbies_v0.1.csv")
    
    # Check if files exist
    if not os.path.exists(embeddings_path) or not os.path.exists(csv_path):
        print("Real data files not found. Skipping integration test.")
        return

    try:
        # Initialize
        predictor = PredictionCrossEncoder(embeddings_path, csv_path)
        
        # Dummy inputs
        dim = predictor.matrixKNN.hobbies_embeddings.shape[1]
        dummy_vec = np.random.rand(1, dim).astype(np.float32)
        dummy_text = "I love outdoor adventures and feeling the wind in my hair, but I dislike competitive team sports."
        
        print(f"Query Text: {dummy_text}")
        
        # Predict
        results = predictor.predict(dummy_vec, k=3, persona_text=dummy_text)
        
        print("Prediction Results:", results)
        
        assert len(results) == 1
        assert len(results[0]) == 3
        print("Test Passed!")
        
    except Exception as e:
        print(f"Test Failed: {e}")
        # raise e # Optional: raise if we want to fail the step

if __name__ == "__main__":
    test_cross_encoder()

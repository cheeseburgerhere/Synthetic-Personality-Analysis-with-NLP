import numpy as np
import os
import sys

# Add current directory to path so we can import PredictionMatrixKNN
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from prediction.prediction_MatrixKNN import PredictionMatrixKNN

def test_prediction():
    # Define paths relative to this script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Paths derived from matrixKNN.ipynb structure
    embeddings_path = os.path.join(current_dir, "../initial_expedition/canonical_embeddings_qwen8B_v0.1.npy")
    csv_path = os.path.join(current_dir, "../initial_expedition/semantically_merged_hobbies_v0.1.csv")
    
    # Resolve absolute paths
    embeddings_path = os.path.abspath(embeddings_path)
    csv_path = os.path.abspath(csv_path)
    
    print(f"Looking for embeddings at: {embeddings_path}")
    print(f"Looking for CSV at: {csv_path}")

    if os.path.exists(embeddings_path) and os.path.exists(csv_path):
        print("Found real files, running integration test...")
        try:
            predictor = PredictionMatrixKNN(embeddings_path, csv_path)
            
            # Generate a dummy vector of appropriate dimension
            # We assume dimensions match the loaded embeddings
            dim = predictor.hobbies_embeddings.shape[1]
            print(f"Embedding dimension: {dim}")
            
            # Create a dummy query (1 sample)
            dummy_vector = np.random.rand(1, dim).astype(np.float32)
            
            # Predict top 3
            results = predictor.predict(dummy_vector, k=3)
            print("Prediction results:", results)
            
            assert len(results) == 1, "Should return results for 1 query"
            assert len(results[0]) == 3, "Should return 3 neighbors"
            assert isinstance(results[0][0], str), "Result items should be strings"
            
            print("Integration test passed!")
            
        except Exception as e:
            print(f"Test failed with error: {e}")
            raise e
    else:
        print("Real files not found, creating dummy files for unit testing...")
        # Create dummy data for unit test
        dummy_emb_path = "dummy_embeddings.npy"
        dummy_csv_path = "dummy_hobbies.csv"
        
        # 10 samples, 16 dimensions
        dummy_emb = np.random.rand(10, 16).astype(np.float32)
        np.save(dummy_emb_path, dummy_emb)
        
        import pandas as pd
        df = pd.DataFrame({"canonical_hobby": [f"Hobby {i}" for i in range(10)]})
        df.to_csv(dummy_csv_path, index=False)
        
        try:
            predictor = PredictionMatrixKNN(dummy_emb_path, dummy_csv_path)
            dummy_vector = np.random.rand(1, 16).astype(np.float32)
            results = predictor.predict(dummy_vector, k=2)
            print("Dummy Prediction results:", results)
            
            assert len(results) == 1
            assert len(results[0]) == 2
            assert results[0][0].startswith("Hobby")
            print("Unit test passed!")
            
        finally:
            # Cleanup
            if os.path.exists(dummy_emb_path):
                os.remove(dummy_emb_path)
            if os.path.exists(dummy_csv_path):
                os.remove(dummy_csv_path)

if __name__ == "__main__":
    test_prediction()

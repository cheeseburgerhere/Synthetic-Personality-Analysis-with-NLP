import numpy as np
from similarity_utils import VectorSimilarity, MatrixKNN, ApproximateKNN

def test_advanced_search():
    print("=== Testing Advanced Search ===")
    
    # Create a small dataset (5 samples, 3 features)
    dataset = [
        [1, 0, 0],  # ID 0
        [0, 1, 0],  # ID 1
        [0, 0, 1],  # ID 2
        [1, 1, 0],  # ID 3: Mix of 0 and 1
        [0.1, 0.9, 0] # ID 4: Close to 1
    ]
    
    # Query: Close to [0, 1, 0]
    query = [0.2, 0.8, 0]
    
    print(f"Dataset Shape: {np.shape(dataset)}")
    print(f"Query: {query}")
    
    # --- Matrix KNN (Cosine) ---
    print("\n1. Matrix KNN (Cosine)")
    knn = MatrixKNN(dataset)
    scores, indices = knn.search_cosine(query, k=3)
    
    print("Top-3 Results:")
    for i in range(len(indices[0])):
        idx = indices[0][i]
        score = scores[0][i]
        print(f"Rank {i+1}: Index {idx} (Data: {dataset[idx]}), Score: {score:.4f}")

    # Expected: 
    # Index 4 ([0.1, 0.9, 0]) should be closest (highest cosine)
    # Index 1 ([0, 1, 0]) next
    # Index 3 ([1, 1, 0]) next

    # --- Matrix KNN (Euclidean) ---
    print("\n2. Matrix KNN (Euclidean)")
    scores_euc, indices_euc = knn.search_euclidean(query, k=3)
    for i in range(len(indices_euc[0])):
        idx = indices_euc[0][i]
        score = scores_euc[0][i]
        print(f"Rank {i+1}: Index {idx}, Distance: {score:.4f}")

    # --- Approximate KNN (sklearn) ---
    print("\n3. Approximate KNN (sklearn defaults)")
    ann = ApproximateKNN(dataset)
    dists_ann, indices_ann = ann.search(query, k=3)
    for i in range(len(indices_ann[0])):
        idx = indices_ann[0][i]
        dist = dists_ann[0][i]
        print(f"Rank {i+1}: Index {idx}, Distance: {dist:.4f}")

    # --- Batch Test ---
    print("\n4. Batch Search Test")
    batch_query = [
        [1, 0, 0], # Should find ID 0 first
        [0, 0, 1]  # Should find ID 2 first
    ]
    b_scores, b_indices = knn.search_cosine(batch_query, k=1)
    print(f"Batch Queries: {len(batch_query)}")
    print(f"Result Indices: {b_indices.flatten().tolist()}")
    
    assert b_indices[0][0] == 0, "Batch Query 0 failed"
    assert b_indices[1][0] == 2, "Batch Query 1 failed"
    print("Batch test passed!")

if __name__ == "__main__":
    test_advanced_search()

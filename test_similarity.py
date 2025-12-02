import numpy as np
from similarity_utils import VectorSimilarity

def test_similarity():
    # Define two simple vectors
    v1 = [1, 0, 0]
    v2 = [0, 1, 0]
    v3 = [1, 1, 0]
    
    print(f"Testing with vectors:\nv1={v1}\nv2={v2}\nv3={v3}\n")

    # 1. Cosine Similarity
    # Orthogonal vectors should have cosine similarity 0
    cos_1_2 = VectorSimilarity.cosine(v1, v2)
    print(f"Cosine(v1, v2): {cos_1_2} (Expected: 0.0)")
    
    # Identical vectors should have cosine similarity 1
    cos_1_1 = VectorSimilarity.cosine(v1, v1)
    print(f"Cosine(v1, v1): {cos_1_1} (Expected: 1.0)")

    # 2. Euclidean Distance
    # Distance between (1,0) and (0,1) is sqrt(1^2 + 1^2) = sqrt(2) ~= 1.414
    euc_1_2 = VectorSimilarity.euclidean(v1, v2)
    print(f"Euclidean(v1, v2): {euc_1_2} (Expected: ~1.414)")

    # 3. Dot Product
    # Orthogonal vectors should have dot product 0
    dot_1_2 = VectorSimilarity.dot_product(v1, v2)
    print(f"Dot(v1, v2): {dot_1_2} (Expected: 0)")
    
    # Dot product of (1,0,0) and (1,1,0) should be 1*1 + 0*1 + 0*0 = 1
    dot_1_3 = VectorSimilarity.dot_product(v1, v3)
    print(f"Dot(v1, v3): {dot_1_3} (Expected: 1)")

    # 4. Manhattan Distance
    # Distance between (1,0) and (0,1) is |1-0| + |0-1| = 2
    man_1_2 = VectorSimilarity.manhattan(v1, v2)
    print(f"Manhattan(v1, v2): {man_1_2} (Expected: 2)")

    # 5. Compute All
    print("\nCompute All (v1, v3):")
    results = VectorSimilarity.compute_all(v1, v3)
    for k, v in results.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    test_similarity()

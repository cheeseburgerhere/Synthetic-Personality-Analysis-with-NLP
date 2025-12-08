import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

class VectorSimilarity:
    """
    A utility class for computing various vector similarity and distance metrics.
    """

    @staticmethod
    def cosine(v1, v2):
        """
        Computes the Cosine Similarity between two vectors.
        Range: [-1, 1] (1 means identical direction).
        """
        v1 = np.array(v1).reshape(1, -1)
        v2 = np.array(v2).reshape(1, -1)
        return cosine_similarity(v1, v2)[0][0]

    @staticmethod
    def euclidean(v1, v2):
        """
        Computes the Euclidean Distance (L2 Norm) between two vectors.
        Range: [0, infinity) (0 means identical).
        """
        v1 = np.array(v1)
        v2 = np.array(v2)
        return np.linalg.norm(v1 - v2)

    @staticmethod
    def dot_product(v1, v2):
        """
        Computes the Dot Product of two vectors.
        """
        v1 = np.array(v1)
        v2 = np.array(v2)
        return np.dot(v1, v2)

    @staticmethod
    def manhattan(v1, v2):
        """
        Computes the Manhattan Distance (L1 Norm) between two vectors.
        """
        v1 = np.array(v1)
        v2 = np.array(v2)
        return np.sum(np.abs(v1 - v2))

    @staticmethod
    def compute_all(v1, v2):
        """
        Computes all available metrics and returns them in a dictionary.
        """
        return {
            "cosine_similarity": VectorSimilarity.cosine(v1, v2),
            "euclidean_distance": VectorSimilarity.euclidean(v1, v2),
            "dot_product": VectorSimilarity.dot_product(v1, v2),
            "manhattan_distance": VectorSimilarity.manhattan(v1, v2)
        }

class MatrixKNN:
    """
    Exact k-NN search using vectorized Matrix Multiplication.
    This is much faster than looping for finding nearest neighbors in a given dataset.
    """
    def __init__(self, data):
        """
        Initialize with a dataset.
        data: numpy array of shape (N_samples, N_features)
        """
        self.data = np.array(data)
        # Pre-compute norms for optimized cosine calculations
        self.norms = np.linalg.norm(self.data, axis=1)
        self.norms[self.norms == 0] = 1e-10  # Avoid division by zero

    def _ensure_2d(self, query):
        """
        Handles variable tensor shapes.
        Ensures query is (N_queries, N_features).
        """
        query = np.array(query)
        if query.ndim == 1:
            return query.reshape(1, -1)
        return query

    def search_cosine(self, query, k=5):
        """
        Vectorized Cosine Similarity Search.
        Returns: (distances, indices)
        Note: Returns cosine distances (1 - similarity) so simpler matches Euclidean.
              Or can return similarities if preferred. Here we return Similarities.
        """
        query = self._ensure_2d(query)
        
        # 1. Compute Dot Product: (B, D) @ (D, N) -> (B, N)
        dot_products = np.dot(query, self.data.T)
        
        # 2. Compute Query Norms: (B,)
        q_norms = np.linalg.norm(query, axis=1)
        q_norms[q_norms == 0] = 1e-10
        
        # 3. Compute Cosine Similarity: Dot / (NormA * NormB)
        # Broadcasting: (B, N) / (B, 1) / (1, N)
        similarities = dot_products / q_norms[:, np.newaxis] / self.norms[np.newaxis, :]
        
        # 4. Get Top-K
        # We want DESCENDING order for similarity.
        # np.argpartition puts the k-th largest element at position -k, 
        # and all larger elements after it. We sort those.
        
        # Handle case where k > N_samples
        n_samples = self.data.shape[0]
        k = min(k, n_samples)
        
        # Indices of top k
        top_k_indices = np.argpartition(similarities, -k, axis=1)[:, -k:]
        
        # Sort the top k indices by similarity score (descending)
        # We take the values, sort them, then rearrange indices
        rows = np.arange(query.shape[0])[:, np.newaxis]
        top_k_values = similarities[rows, top_k_indices]
        
        # Sort indices within the top k (argsort is ascending, so flip)
        sorted_internal_indices = np.argsort(top_k_values, axis=1)[:, ::-1]
        
        final_indices = top_k_indices[rows, sorted_internal_indices]
        final_scores = top_k_values[rows, sorted_internal_indices]
        
        return final_scores, final_indices

    def search_euclidean(self, query, k=5):
        """
        Vectorized Euclidean Distance Search.
        Returns: (distances, indices)
        """
        query = self._ensure_2d(query)
        
        # Optimized Euclidean Distance: |A - B|^2 = |A|^2 + |B|^2 - 2A.B
        # This allows matrix mult speedup instead of expanding (A-B)
        
        d_squared = np.sum(self.data**2, axis=1)[np.newaxis, :] + \
                    np.sum(query**2, axis=1)[:, np.newaxis] - \
                    2 * np.dot(query, self.data.T)
                    
        # Numerical stability clips
        d_squared = np.maximum(d_squared, 0)
        distances = np.sqrt(d_squared)
        
        n_samples = self.data.shape[0]
        k = min(k, n_samples)
        
        # Smallest distances are best
        top_k_indices = np.argpartition(distances, k, axis=1)[:, :k]
        
        rows = np.arange(query.shape[0])[:, np.newaxis]
        top_k_values = distances[rows, top_k_indices]
        
        sorted_internal_indices = np.argsort(top_k_values, axis=1)
        
        final_indices = top_k_indices[rows, sorted_internal_indices]
        final_scores = top_k_values[rows, sorted_internal_indices]
        
        return final_scores, final_indices


class ApproximateKNN:
    """
    Approximate k-NN using Tree algorithms (KDTree/BallTree).
    Wraps sklearn.neighbors.NearestNeighbors.
    This is generally faster for very low dimensional data or useful for specific metrics,
    though for high-dim embeddings, Matrix-KNN is often competitive or faster than trees.
    """
    def __init__(self, data, algorithm='auto'):
        self.data = np.array(data)
        self.model = NearestNeighbors(algorithm=algorithm)
        self.model.fit(self.data)
        
    def search(self, query, k=5):
        """
        Returns (distances, indices)
        Note: sklearn defaults to Euclidean distance (Minkowski p=2).
        """
        query = np.array(query)
        if query.ndim == 1:
            query = query.reshape(1, -1)
            
        distances, indices = self.model.kneighbors(query, n_neighbors=k)
        return distances, indices

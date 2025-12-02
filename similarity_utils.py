import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

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

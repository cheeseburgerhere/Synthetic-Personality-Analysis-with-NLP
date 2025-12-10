import numpy as np
import pandas as pd
import lzma
from sklearn.cluster import AgglomerativeClustering
from concurrent.futures import ThreadPoolExecutor
import logging
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(__file__), 'logs', 'clustering.log')),
        logging.StreamHandler()
    ]
)

import zlib

class NCDClustering:
    def __init__(self, compressor='zlib'):
        if compressor == 'lzma':
            self.compressor = lzma
            self.func = self.compressor.compress
        elif compressor == 'zlib':
            self.compressor = zlib
            self.func = self.compressor.compress
        else:
            raise ValueError("Unknown compressor")
    
    def get_compression_len(self, text):
        if not isinstance(text, bytes):
            text = text.encode('utf-8')
        return len(self.func(text))

    def ncd(self, x, y):
        """
        Compute Normalized Compression Distance between two strings x and y.
        NCD(x,y) = (C(xy) - min(C(x), C(y))) / max(C(x), C(y))
        """
        cx = self.get_compression_len(x)
        cy = self.get_compression_len(y)
        
        # Concatenate x and y
        xy = x + " " + y # Space separator helps standard compressors
        cxy = self.get_compression_len(xy)
        
        return (cxy - min(cx, cy)) / max(cx, cy)

    def compute_distance_matrix(self, texts):
        n = len(texts)
        dist_matrix = np.zeros((n, n))
        
        logging.info(f"Computing NCD matrix for {n} items...")
        
        # This can be slow O(N^2), optimizing with parallel execution if needed
        # For simplicity and clarity in "research code", we stick to straightforward nested loops
        # but usage of ThreadPoolExecutor for rows can speed it up.
        
        # Precompute individual compression lengths
        c_lens = [self.get_compression_len(t) for t in texts]
        
        for i in range(n):
            for j in range(i + 1, n):
                # We can compute just one triangle
                x = texts[i]
                y = texts[j]
                
                cx = c_lens[i]
                cy = c_lens[j]
                
                # Combine
                xy = x + " " + y
                cxy = self.get_compression_len(xy)
                
                score = (cxy - min(cx, cy)) / max(cx, cy)
                dist_matrix[i, j] = score
                dist_matrix[j, i] = score
                
        logging.info("Distance matrix computed.")
        return dist_matrix

    def cluster(self, texts, distance_threshold=0.5):
        """
        Clusters texts using Agglomerative Clustering with precomputed NCD matrix.
        """
        dist_matrix = self.compute_distance_matrix(texts)
        
        clustering = AgglomerativeClustering(
            n_clusters=None,
            metric='precomputed',
            linkage='complete',
            distance_threshold=distance_threshold
        )
        
        labels = clustering.fit_predict(dist_matrix)
        
        # Group results
        clusters = {}
        for text, label in zip(texts, labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(text)
            
        return clusters

def main():
    # Sanity check
    tests = ["soccer", "playing soccer", "football", "cooking", "baking", "finance"]
    # Expected: (soccer, playing soccer, football), (cooking, baking), (finance)
    
    ncd = NCDClustering()
    clusters = ncd.cluster(tests, distance_threshold=0.8) # Higher threshold for NCD as it ranges 0-1+
    
    print("Clusters found:")
    for label, items in clusters.items():
        print(f"Cluster {label}: {items}")

if __name__ == "__main__":
    main()

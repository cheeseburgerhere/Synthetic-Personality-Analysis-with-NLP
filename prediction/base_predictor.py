from abc import ABC, abstractmethod

class BasePredictor(ABC):
    """
    Abstract base class for all hobby predictors.
    """
    
    @abstractmethod
    def predict(self, vectors, k=5, **kwargs):
        """
        Predict hobbies for given inputs.
        
        Args:
            vectors (np.array): Query vectors of shape (N, D) or (D,).
            k (int): Number of nearest neighbors to return.
            **kwargs: Additional arguments for future flexibility (e.g., original_text).
            
        Returns:
            list of list of str: Predicted hobbies for each query vector.
        """
        pass

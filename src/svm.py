# Linear SVM from scratch (hinge loss + SGD)
import numpy as np
import pickle
import json
from abc import ABC, abstractmethod


class SVMBase(ABC):
    """
    Base class for SVM implementations.
    """
    def __init__(self, lr=0.01, C=1.0, n_iter=1000):
        """
        Parameters:
        -----------
        lr : float
            Learning rate for gradient descent
        C : float
            Regularization parameter (larger C = less regularization)
        n_iter : int
            Number of iterations for training
        """
        self.lr = lr
        self.C = C
        self.n_iter = n_iter
        self.w = None
        self.b = None
        self.classes_ = None
    
    def _hinge_loss(self, X, y):
        """
        Compute hinge loss with L2 regularization.
        Loss = 1/m * Σ max(0, 1 - y*(w^T*x + b)) + λ||w||^2
        """
        # TODO: Implementasi hinge loss
        pass
    
    def _compute_gradients(self, X, y):
        """
        Compute gradients for SVM.
        For samples where y*(w^T*x + b) < 1:
            dw = w - C*y*x
            db = -C*y
        Otherwise:
            dw = w
            db = 0
        """
        # TODO: Implementasi gradient computation
        pass
    
    def _fit_binary(self, X, y):
        """
        Train binary SVM classifier.
        Assumes y is in {-1, +1}
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Binary target values in {-1, +1}
        """
        # TODO: Implementasi binary SVM training
        # 1. Initialize w and b
        # 2. For each iteration:
        #    - Compute gradients
        #    - Update w and b using gradient descent
        pass
    
    def decision_function(self, X):
        """
        Compute decision function: w^T * x + b
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Samples
            
        Returns:
        --------
        scores : array of shape (n_samples,)
            Decision scores
        """
        # TODO: return X @ self.w + self.b
        pass
    
    @abstractmethod
    def fit(self, X, y):
        """Train the SVM classifier. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def predict(self, X):
        """Predict class labels. Must be implemented by subclasses."""
        pass
    
    def save(self, filepath):
        """Save model to file."""
        if filepath.endswith('.pkl'):
            with open(filepath, 'wb') as f:
                pickle.dump(self, f)
        elif filepath.endswith('.txt') or filepath.endswith('.json'):
            # Each subclass should override this for specific attributes
            model_dict = self._get_save_dict()
            with open(filepath, 'w') as f:
                json.dump(model_dict, f, indent=2)
        else:
            raise ValueError("Filepath must end with .pkl, .txt, or .json")
    
    def _get_save_dict(self):
        """Get dictionary for JSON serialization. Override in subclasses if needed."""
        return {
            'lr': self.lr,
            'C': self.C,
            'n_iter': self.n_iter,
            'w': self.w.tolist() if self.w is not None else None,
            'b': float(self.b) if self.b is not None else None,
            'classes': self.classes_.tolist() if self.classes_ is not None else None
        }
    
    @classmethod
    def load(cls, filepath):
        """Load model from file."""
        if filepath.endswith('.pkl'):
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        elif filepath.endswith('.txt') or filepath.endswith('.json'):
            with open(filepath, 'r') as f:
                model_dict = json.load(f)
            model = cls(
                lr=model_dict['lr'],
                C=model_dict['C'],
                n_iter=model_dict['n_iter']
            )
            model.w = np.array(model_dict['w']) if model_dict['w'] is not None else None
            model.b = model_dict['b']
            model.classes_ = np.array(model_dict['classes']) if model_dict['classes'] is not None else None
            return model
        else:
            raise ValueError("Filepath must end with .pkl, .txt, or .json")


class SVMOneVsAll(SVMBase):
    """
    SVM with One-vs-All (OvA) / One-vs-Rest (OvR) multiclass strategy.
    
    For K classes, trains K binary classifiers.
    Each classifier distinguishes one class from all others.
    Prediction: choose class with highest decision score.
    """
    def __init__(self, lr=0.01, C=1.0, n_iter=1000):
        super().__init__(lr, C, n_iter)
        self.classifiers_ = []  # List of (w, b) for each class
    
    def fit(self, X, y):
        """
        Train OvA SVM classifier.
        """
        # TODO: Implementasi OvA training
        # 1. Get unique classes
        # 2. For each class k:
        #    - Create binary labels: y_k = +1 if y==k else -1
        #    - Train binary SVM
        #    - Store (w_k, b_k)
        pass
    
    def decision_function(self, X):
        """
        Compute decision scores for all classes.
        """
        # TODO: Compute w_k^T * x + b_k for each classifier k
        pass
    
    def predict(self, X):
        """
        Predict class labels using OvA strategy.
        Choose class with highest decision score.
        
        """
        # TODO: Return argmax of decision scores
        pass
    
    def _get_save_dict(self):
        """Override to save all classifiers."""
        base_dict = super()._get_save_dict()
        base_dict['classifiers'] = [
            {'w': w.tolist(), 'b': float(b)} 
            for w, b in self.classifiers_
        ] if self.classifiers_ else []
        return base_dict


class SVMOneVsOne(SVMBase):
    """
    SVM with One-vs-One (OvO) multiclass strategy.
    
    For K classes, trains K*(K-1)/2 binary classifiers.
    Each classifier distinguishes between two classes.
    Prediction: voting scheme (class with most wins).
    """
    def __init__(self, lr=0.01, C=1.0, n_iter=1000):
        super().__init__(lr, C, n_iter)
        self.classifiers_ = []  # List of ((class_i, class_j), (w, b))
        self.class_pairs_ = []
    
    def fit(self, X, y):
        """
        Train OvO SVM classifier.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values
        """
        # TODO: Implementasi OvO training
        # 1. Get unique classes
        # 2. For each pair (class_i, class_j) where i < j:
        #    - Select samples with y in {class_i, class_j}
        #    - Create binary labels: +1 for class_i, -1 for class_j
        #    - Train binary SVM
        #    - Store ((class_i, class_j), (w, b))
        pass
    
    def predict(self, X):
        """
        Predict class labels using OvO voting.
        
        For each sample:
        - Get predictions from all K*(K-1)/2 classifiers
        - Count votes for each class
        - Return class with most votes
        
        Returns:
        --------
        y_pred : array of shape (n_samples,)
            Predicted class labels
        """
        # TODO: Implementasi voting scheme
        pass
    
    def _get_save_dict(self):
        """Override to save all pairwise classifiers."""
        base_dict = super()._get_save_dict()
        base_dict['classifiers'] = [
            {
                'classes': [int(c1), int(c2)],
                'w': w.tolist(),
                'b': float(b)
            }
            for (c1, c2), (w, b) in self.classifiers_
        ] if self.classifiers_ else []
        return base_dict


class SVMDAG(SVMBase):
    """
    SVM with Directed Acyclic Graph (DAG) multiclass strategy.
    
    Similar to OvO but uses a decision DAG for efficient prediction.
    Trains K*(K-1)/2 binary classifiers but organizes them in a DAG structure.
    Prediction: O(K-1) evaluations instead of O(K²) for OvO.
    
    DAG structure:
    - Root: compare class 1 vs class K
    - Eliminate loser, continue with remaining classes
    - Leaf: final class prediction
    """
    def __init__(self, lr=0.01, C=1.0, n_iter=1000):
        super().__init__(lr, C, n_iter)
        self.classifiers_ = {}  # Dict: (class_i, class_j) -> (w, b)
    
    def fit(self, X, y):
        """
        Train DAGSVM classifier.
        
        Training is same as OvO: train all pairwise classifiers.
        Difference is in prediction (DAG traversal).
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values
        """
        # TODO: Implementasi DAGSVM training (sama seperti OvO)
        # 1. Get unique classes
        # 2. For each pair (class_i, class_j) where i < j:
        #    - Select samples with y in {class_i, class_j}
        #    - Train binary SVM
        #    - Store in dict: (class_i, class_j) -> (w, b)
        pass
    
    def _predict_sample_dag(self, x, classes):
        """
        Predict single sample using DAG traversal.
        
        Algorithm:
        1. Start with all K classes
        2. While len(classes) > 1:
           - Compare first and last class in list
           - Eliminate loser
           - Continue with remaining classes
        3. Return final class
        """
        # TODO: Implementasi DAG traversal
        pass
    
    def predict(self, X):
        """
        Predict class labels using DAG strategy.
        
        Returns:
        --------
        y_pred : array of shape (n_samples,)
            Predicted class labels
        """
        # TODO: Predict each sample using DAG
        # return np.array([self._predict_sample_dag(x, self.classes_) for x in X])
        pass
    
    def _get_save_dict(self):
        """Override to save DAG classifiers."""
        base_dict = super()._get_save_dict()
        base_dict['classifiers'] = {
            f"{int(c1)}_{int(c2)}": {
                'w': w.tolist(),
                'b': float(b)
            }
            for (c1, c2), (w, b) in self.classifiers_.items()
        } if self.classifiers_ else {}
        return base_dict


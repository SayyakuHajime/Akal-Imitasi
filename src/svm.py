# Linear SVM from scratch (hinge loss + SGD)
import numpy as np
import pickle
import json
from abc import ABC, abstractmethod


class _BinarySVM:
    """
    Helper class for binary SVM classification.
    Not meant to be used directly - used internally by multiclass strategies.
    """
    def __init__(self, lr=0.01, C=1.0, n_iter=1000):
        self.lr = lr
        self.C = C
        self.n_iter = n_iter
        self.w = None
        self.b = None
    
    def _hinge_loss(self, X, y):
        """Compute hinge loss with L2 regularization."""
        m = X.shape[0]
        distances = 1 - y * (X @ self.w + self.b)
        hinge_loss = np.maximum(0, distances)
        reg_loss = 0.5 * np.sum(self.w ** 2)
        return np.sum(hinge_loss) / m + reg_loss
    
    def _compute_gradients(self, X, y):
        """Compute gradients for SVM."""
        m = X.shape[0]
        distances = y * (X @ self.w + self.b)
        
        dw = self.w.copy()
        db = 0
        
        for idx, d in enumerate(distances):
            if d < 1:
                dw -= self.C * y[idx] * X[idx]
                db -= self.C * y[idx]
        
        return dw / m, db / m
    
    def fit(self, X, y):
        """Train binary SVM. Assumes y in {-1, +1}."""
        m, n = X.shape
        self.w = np.zeros(n)
        self.b = 0
        
        for iteration in range(self.n_iter):
            dw, db = self._compute_gradients(X, y)
            self.w -= self.lr * dw
            self.b -= self.lr * db
    
    def decision_function(self, X):
        """Compute decision function: w^T * x + b."""
        return X @ self.w + self.b


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
        X = np.array(X) if not isinstance(X, np.ndarray) else X
        y = np.array(y) if not isinstance(y, np.ndarray) else y
        
        self.classes_ = np.unique(y)
        self.classifiers_ = []
        
        # Train K binary classifiers
        for class_label in self.classes_:
            # Create binary labels: +1 for current class, -1 for others
            y_binary = np.where(y == class_label, 1, -1)
            
            # Create and train binary SVM
            clf = _BinarySVM(lr=self.lr, C=self.C, n_iter=self.n_iter)
            clf.fit(X, y_binary)
            
            # Store classifier
            self.classifiers_.append((clf.w, clf.b))
    
    def decision_function(self, X):
        """
        Compute decision scores for all classes.
        """
        X = np.array(X) if not isinstance(X, np.ndarray) else X
        
        # Compute decision scores for each classifier
        scores = np.zeros((X.shape[0], len(self.classes_)))
        
        for i, (w, b) in enumerate(self.classifiers_):
            scores[:, i] = X @ w + b
        
        return scores
    
    def predict(self, X):
        """
        Predict class labels using OvA strategy.
        Choose class with highest decision score.
        """
        scores = self.decision_function(X)
        predictions = np.argmax(scores, axis=1)
        return self.classes_[predictions]
    
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
        X = np.array(X) if not isinstance(X, np.ndarray) else X
        y = np.array(y) if not isinstance(y, np.ndarray) else y
        
        self.classes_ = np.unique(y)
        self.classifiers_ = []
        self.class_pairs_ = []
        
        # Train K*(K-1)/2 binary classifiers
        for i in range(len(self.classes_)):
            for j in range(i + 1, len(self.classes_)):
                class_i, class_j = self.classes_[i], self.classes_[j]
                
                # Select samples from these two classes
                mask = (y == class_i) | (y == class_j)
                X_subset = X[mask]
                y_subset = y[mask]
                
                # Create binary labels: +1 for class_i, -1 for class_j
                y_binary = np.where(y_subset == class_i, 1, -1)
                
                # Train binary SVM
                clf = _BinarySVM(lr=self.lr, C=self.C, n_iter=self.n_iter)
                clf.fit(X_subset, y_binary)
                
                # Store classifier and class pair
                self.classifiers_.append(((class_i, class_j), (clf.w, clf.b)))
                self.class_pairs_.append((class_i, class_j))
    
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
        X = np.array(X) if not isinstance(X, np.ndarray) else X
        n_samples = X.shape[0]
        predictions = []
        
        for x in X:
            # Vote for each class
            votes = {class_label: 0 for class_label in self.classes_}
            
            for (class_i, class_j), (w, b) in self.classifiers_:
                # Predict with binary classifier
                decision = np.dot(w, x) + b
                
                # Vote: positive decision → class_i, negative → class_j
                if decision > 0:
                    votes[class_i] += 1
                else:
                    votes[class_j] += 1
            
            # Return class with most votes
            predicted_class = max(votes, key=votes.get)
            predictions.append(predicted_class)
        
        return np.array(predictions)
    
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
        X = np.array(X) if not isinstance(X, np.ndarray) else X
        y = np.array(y) if not isinstance(y, np.ndarray) else y
        
        self.classes_ = np.unique(y)
        self.classifiers_ = {}
        
        # Train K*(K-1)/2 binary classifiers (same as OvO)
        for i in range(len(self.classes_)):
            for j in range(i + 1, len(self.classes_)):
                class_i, class_j = self.classes_[i], self.classes_[j]
                
                # Select samples from these two classes
                mask = (y == class_i) | (y == class_j)
                X_subset = X[mask]
                y_subset = y[mask]
                
                # Create binary labels: +1 for class_i, -1 for class_j
                y_binary = np.where(y_subset == class_i, 1, -1)
                
                # Train binary SVM
                clf = _BinarySVM(lr=self.lr, C=self.C, n_iter=self.n_iter)
                clf.fit(X_subset, y_binary)
                
                # Store in dictionary with tuple key
                self.classifiers_[(class_i, class_j)] = (clf.w, clf.b)
    
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
        classes = list(classes)  # Make a copy
        
        while len(classes) > 1:
            # Compare first and last class
            class_i = classes[0]
            class_j = classes[-1]
            
            # Get classifier (order matters: (i, j) where i < j)
            if class_i < class_j:
                key = (class_i, class_j)
                sign = 1
            else:
                key = (class_j, class_i)
                sign = -1  # Reverse sign if order is swapped
            
            w, b = self.classifiers_[key]
            decision = sign * (np.dot(w, x) + b)
            
            # Eliminate loser
            if decision > 0:
                # class_i wins, eliminate class_j
                classes.pop()
            else:
                # class_j wins, eliminate class_i
                classes.pop(0)
        
        return classes[0]
    
    def predict(self, X):
        """
        Predict class labels using DAG strategy.
        
        Returns:
        --------
        y_pred : array of shape (n_samples,)
            Predicted class labels
        """
        X = np.array(X) if not isinstance(X, np.ndarray) else X
        return np.array([self._predict_sample_dag(x, self.classes_) for x in X])
    
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


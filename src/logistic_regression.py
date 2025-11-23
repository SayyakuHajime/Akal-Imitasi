# Logistic Regression from scratch using Stochastic Gradient Ascent
import numpy as np
import pickle
import json

class LogisticRegressionScratch:
    """
    Logistic Regression classifier using Stochastic Gradient Ascent (SGA).
    Supports binary and multiclass classification (One-vs-Rest).
    """
    def __init__(self, lr=0.01, n_iter=1000, l2=0.0, batch_size=1):
        """
        Parameters:
        -----------
        lr : float
            Learning rate for gradient ascent
        n_iter : int
            Number of epochs/iterations for training
        l2 : float
            L2 regularization parameter (Ridge)
        batch_size : int or 'full'
            Batch size for updates:
            - 1: Stochastic Gradient Ascent (update per sample)
            - n (1 < n < m): Mini-batch Gradient Ascent
            - 'full' or m: Batch Gradient Ascent
        """
        self.lr = lr
        self.n_iter = n_iter
        self.l2 = l2
        self.batch_size = batch_size
        self.w = None
        self.b = None
        self.loss_history = []
        self.classes_ = None
        self.is_multiclass = False
    
    def _sigmoid(self, z):
        """
        Sigmoid activation function.
        σ(z) = 1 / (1 + e^(-z))
        """
        # Clip to prevent overflow
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def _log_likelihood(self, X, y, y_pred):
        """
        Compute log-likelihood (for gradient ASCENT).
        
        Log-likelihood: L(w) = Σ[y*log(h) + (1-y)*log(1-h)]
        With L2 regularization: L(w) - λ/2 * ||w||^2
        
        Note: We MAXIMIZE this (ascent), not minimize.
        """
        # TODO: Implementasi log-likelihood
        # Clip predictions to avoid log(0)
        # eps = 1e-15
        # y_pred = np.clip(y_pred, eps, 1 - eps)
        # ll = np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
        # if self.l2 > 0:
        #     ll -= (self.l2 / 2) * np.sum(self.w ** 2)
        # return ll
        pass
    
    def _compute_gradients_ascent(self, X, y, y_pred):
        """
        Compute gradients for ASCENT (maximize log-likelihood).
        
        Gradient of log-likelihood w.r.t. w:
        ∂L/∂w = X^T * (y - h(x)) - λ*w
        
        For ASCENT: w = w + α * gradient (note the + sign)
        For DESCENT: w = w - α * gradient (note the - sign)
        
        Parameters:
        -----------
        X : array of shape (batch_size, n_features)
        y : array of shape (batch_size,)
        y_pred : array of shape (batch_size,)
        
        Returns:
        --------
        grad_w : gradient for weights
        grad_b : gradient for bias
        """
        # TODO: Implementasi gradient computation untuk ASCENT
        # m = X.shape[0]
        # grad_w = (1/m) * X.T @ (y - y_pred)
        # if self.l2 > 0:
        #     grad_w -= (self.l2 / m) * self.w
        # grad_b = (1/m) * np.sum(y - y_pred)
        # return grad_w, grad_b
        pass
    
    def fit(self, X, y):
        """
        Train logistic regression using Stochastic Gradient Ascent.
        
        Algorithm:
        1. Initialize weights w and bias b
        2. For each epoch (n_iter times):
           a. Shuffle training data
           b. For each batch:
              - Compute predictions
              - Compute gradients (ascent direction)
              - Update: w = w + lr * grad_w (ASCENT, note the +)
                        b = b + lr * grad_b
        3. Store log-likelihood history for visualization (bonus)
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values (0 or 1 for binary)
        """
        # TODO: Implementasi Stochastic Gradient Ascent
        # 1. Convert to numpy arrays
        # X = np.array(X)
        # y = np.array(y)
        # m, n = X.shape
        
        # 2. Initialize weights and bias
        # self.w = np.zeros(n)
        # self.b = 0
        
        # 3. Determine batch size
        # if self.batch_size == 'full':
        #     batch_size = m
        # else:
        #     batch_size = min(self.batch_size, m)
        
        # 4. Training loop
        # for epoch in range(self.n_iter):
        #     # Shuffle data for stochastic behavior
        #     indices = np.random.permutation(m)
        #     X_shuffled = X[indices]
        #     y_shuffled = y[indices]
        #     
        #     # Mini-batch/stochastic updates
        #     for i in range(0, m, batch_size):
        #         X_batch = X_shuffled[i:i+batch_size]
        #         y_batch = y_shuffled[i:i+batch_size]
        #         
        #         # Forward pass
        #         y_pred = self._sigmoid(X_batch @ self.w + self.b)
        #         
        #         # Compute gradients (ASCENT)
        #         grad_w, grad_b = self._compute_gradients_ascent(X_batch, y_batch, y_pred)
        #         
        #         # Update parameters (ASCENT: w = w + lr * grad)
        #         self.w += self.lr * grad_w
        #         self.b += self.lr * grad_b
        #     
        #     # Compute and store log-likelihood for entire dataset
        #     y_pred_full = self._sigmoid(X @ self.w + self.b)
        #     ll = self._log_likelihood(X, y, y_pred_full)
        #     self.loss_history.append(ll)
        pass
    
    def predict_proba(self, X):
        """
        Predict class probabilities.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Samples
            
        Returns:
        --------
        proba : array of shape (n_samples,) or (n_samples, n_classes)
            Predicted probabilities
        """
        # TODO: return self._sigmoid(X @ self.w + self.b)
        pass
    
    def predict(self, X, threshold=0.5):
        """
        Predict class labels.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Samples
        threshold : float
            Decision threshold (default 0.5)
            
        Returns:
        --------
        y_pred : array of shape (n_samples,)
            Predicted class labels
        """
        # TODO: Apply threshold to probabilities
        # return (self.predict_proba(X) >= threshold).astype(int)
        pass
    
    def save(self, filepath):
        """
        Save model to file.
        """
        if filepath.endswith('.pkl'):
            with open(filepath, 'wb') as f:
                pickle.dump(self, f)
        elif filepath.endswith('.txt') or filepath.endswith('.json'):
            model_dict = {
                'lr': self.lr,
                'n_iter': self.n_iter,
                'l2': self.l2,
                'batch_size': self.batch_size,
                'w': self.w.tolist() if self.w is not None else None,
                'b': float(self.b) if self.b is not None else None,
                'classes': self.classes_.tolist() if self.classes_ is not None else None
            }
            with open(filepath, 'w') as f:
                json.dump(model_dict, f, indent=2)
        else:
            raise ValueError("Filepath must end with .pkl, .txt, or .json")
    
    @classmethod
    def load(cls, filepath):
        """
        Load model from file.
        """
        if filepath.endswith('.pkl'):
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        elif filepath.endswith('.txt') or filepath.endswith('.json'):
            with open(filepath, 'r') as f:
                model_dict = json.load(f)
            model = cls(
                lr=model_dict['lr'],
                n_iter=model_dict['n_iter'],
                l2=model_dict['l2'],
                batch_size=model_dict.get('batch_size', 1)
            )
            model.w = np.array(model_dict['w']) if model_dict['w'] is not None else None
            model.b = model_dict['b']
            model.classes_ = np.array(model_dict['classes']) if model_dict['classes'] is not None else None
            return model
        else:
            raise ValueError("Filepath must end with .pkl, .txt, or .json")


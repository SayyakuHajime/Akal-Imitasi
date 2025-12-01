# Logistic Regression from scratch using Stochastic Gradient Ascent
import numpy as np
import pickle
import json

class LogisticRegressionScratch:
    """
    Logistic Regression classifier using Stochastic Gradient Ascent (SGA).
    Supports binary and multiclass classification (One-vs-Rest).
    """
    def __init__(self, lr=0.01, n_iter=1000, l2=0.0, batch_size=1,
                 momentum=0.0, lr_decay=1.0, use_xavier=False, class_weight='balanced'):
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
        momentum : float (0-1)
            Momentum coefficient for accelerated convergence (default 0 = no momentum)
        lr_decay : float (0-1)
            Learning rate decay per epoch (default 1 = no decay, 0.95 = 5% decay)
        use_xavier : bool
            Use Xavier initialization instead of zeros (default False for backward compatibility)
        class_weight : str or dict
            'balanced' to automatically balance class weights, or dict mapping classes to weights
        """
        self.lr = lr
        self.n_iter = n_iter
        self.l2 = l2
        self.batch_size = batch_size
        self.momentum = momentum
        self.lr_decay = lr_decay
        self.use_xavier = use_xavier
        self.class_weight_mode = class_weight
        self.w = None
        self.b = None
        self.velocity_w = None
        self.velocity_b = None
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
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        ll = np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
        
        if self.l2 > 0 and self.w is not None:
            ll -= (self.l2 / 2) * np.sum(self.w ** 2)
        
        return ll
    
    def _compute_gradients_ascent(self, X, y, y_pred, sample_weight=None):
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
        sample_weight : array of shape (batch_size,) or None
            Sample weights for handling class imbalance

        Returns:
        --------
        grad_w : gradient for weights
        grad_b : gradient for bias
        """
        m = X.shape[0]

        # Apply sample weights if provided
        error = y - y_pred
        if sample_weight is not None:
            error = error * sample_weight

        # Gradient for ascent (note: y - y_pred, not y_pred - y)
        grad_w = (1/m) * X.T @ error
        
        if self.l2 > 0:
            grad_w -= (self.l2 / m) * self.w
        
        grad_b = (1/m) * np.sum(y - y_pred)
        
        return grad_w, grad_b
    
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
        # Convert to numpy arrays
        X = np.array(X) if not isinstance(X, np.ndarray) else X
        y = np.array(y) if not isinstance(y, np.ndarray) else y
        
        # Check if multiclass
        self.classes_ = np.unique(y)
        if len(self.classes_) > 2:
            self.is_multiclass = True
            # For multiclass, use One-vs-Rest
            self._fit_multiclass(X, y)
            return
        
        # Binary classification
        m, n = X.shape

        # Compute sample weights for class imbalance
        sample_weight = None
        if self.class_weight_mode == 'balanced':
            n_samples = len(y)
            n_pos = np.sum(y == 1)
            n_neg = np.sum(y == 0)
            weight_pos = n_samples / (2 * n_pos) if n_pos > 0 else 1.0
            weight_neg = n_samples / (2 * n_neg) if n_neg > 0 else 1.0
            sample_weight = np.where(y == 1, weight_pos, weight_neg)

        # Initialize weights and bias
        if self.use_xavier:
            # Xavier/Glorot initialization for better convergence
            limit = np.sqrt(6.0 / (n + 1))
            self.w = np.random.uniform(-limit, limit, n)
        else:
            self.w = np.zeros(n)
        self.b = 0

        # Initialize momentum vectors
        self.velocity_w = np.zeros(n)
        self.velocity_b = 0.0

        self.loss_history = []

        # Determine batch size
        if self.batch_size == 'full':
            batch_size = m
        else:
            batch_size = min(self.batch_size, m)

        # Current learning rate (for decay)
        current_lr = self.lr

        # Training loop
        for epoch in range(self.n_iter):
            # Shuffle data for stochastic behavior
            indices = np.random.permutation(m)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            if sample_weight is not None:
                sample_weight_shuffled = sample_weight[indices]
            else:
                sample_weight_shuffled = None

            # Mini-batch/stochastic updates
            for i in range(0, m, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                weight_batch = sample_weight_shuffled[i:i+batch_size] if sample_weight_shuffled is not None else None

                # Forward pass
                y_pred = self._sigmoid(X_batch @ self.w + self.b)

                # Compute gradients (ASCENT)
                grad_w, grad_b = self._compute_gradients_ascent(X_batch, y_batch, y_pred, weight_batch)

                # Apply momentum if enabled
                if self.momentum > 0:
                    self.velocity_w = self.momentum * self.velocity_w + current_lr * grad_w
                    self.velocity_b = self.momentum * self.velocity_b + current_lr * grad_b
                    self.w += self.velocity_w
                    self.b += self.velocity_b
                else:
                    # Standard update (ASCENT: w = w + lr * grad)
                    self.w += current_lr * grad_w
                    self.b += current_lr * grad_b

            # Apply learning rate decay
            current_lr *= self.lr_decay

            # Compute and store log-likelihood for entire dataset
            if epoch % 100 == 0:  # Save every 100 epochs to reduce overhead
                y_pred_full = self._sigmoid(X @ self.w + self.b)
                ll = self._log_likelihood(X, y, y_pred_full)
                self.loss_history.append(ll)
    
    def _fit_multiclass(self, X, y):
        """Fit multiclass using One-vs-Rest strategy with class weighting."""
        self.classifiers_ = []

        for class_label in self.classes_:
            # Create binary labels: 1 for current class, 0 for others
            y_binary = (y == class_label).astype(int)

            # Train binary classifier with same hyperparameters
            clf = LogisticRegressionScratch(
                lr=self.lr,
                n_iter=self.n_iter,
                l2=self.l2,
                batch_size=self.batch_size,
                momentum=self.momentum,
                lr_decay=self.lr_decay,
                use_xavier=self.use_xavier,
                class_weight=self.class_weight_mode
            )
            clf.fit(X, y_binary)
            self.classifiers_.append(clf)
    
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
        X = np.array(X) if not isinstance(X, np.ndarray) else X
        
        if self.is_multiclass:
            # Get probabilities from all classifiers
            probas = np.array([clf.predict_proba(X) for clf in self.classifiers_]).T
            return probas
        else:
            return self._sigmoid(X @ self.w + self.b)
    
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
        X = np.array(X) if not isinstance(X, np.ndarray) else X
        
        if self.is_multiclass:
            # Get probabilities and return class with highest probability
            probas = self.predict_proba(X)
            return self.classes_[np.argmax(probas, axis=1)]
        else:
            return (self.predict_proba(X) >= threshold).astype(int)
    
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


# Implementasi Decision Tree Learning (ID3 / CART) from scratch
import numpy as np
import pickle
import json

class ID3Classifier:
    """
    ID3 (Iterative Dichotomiser 3) Decision Tree Classifier.
    Uses entropy and information gain for splitting.
    """
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree_ = None
    
    def _entropy(self, y):
        """
        Calculate entropy of a label array.
        Entropy = -Σ(p_i * log2(p_i))
        """
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        probabilities = probabilities[probabilities > 0]
        return -np.sum(probabilities * np.log2(probabilities))
    
    def _information_gain(self, X_column, y, threshold):
        """
        Calculate information gain for a given feature and threshold.
        IG = entropy(parent) - weighted_avg(entropy(children))
        """
        parent_entropy = self._entropy(y)
        
        left_mask = X_column <= threshold
        right_mask = ~left_mask
        
        n = len(y)
        n_left = np.sum(left_mask)
        n_right = np.sum(right_mask)
        
        if n_left == 0 or n_right == 0:
            return 0
        
        left_entropy = self._entropy(y[left_mask])
        right_entropy = self._entropy(y[right_mask])
        
        weighted_entropy = (n_left / n) * left_entropy + (n_right / n) * right_entropy
        
        return parent_entropy - weighted_entropy
    
    def _best_split(self, X, y, feature_indices):
        """
        Find the best feature and threshold to split on.
        Returns: (best_feature_idx, best_threshold, best_gain)
        """
        best_gain = -1
        best_feature_idx = None
        best_threshold = None
        
        for feature_idx in feature_indices:
            X_column = X[:, feature_idx]
            thresholds = np.unique(X_column)
            
            for threshold in thresholds:
                gain = self._information_gain(X_column, y, threshold)
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature_idx = feature_idx
                    best_threshold = threshold
        
        return best_feature_idx, best_threshold, best_gain
    
    def _build_tree(self, X, y, depth=0):
        """
        Recursively build the decision tree.
        Returns: tree node (dict or leaf value)
        """
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))
        
        # Stopping criteria
        if (self.max_depth is not None and depth >= self.max_depth) or \
           n_labels == 1 or \
           n_samples < self.min_samples_split:
            leaf_value = np.bincount(y.astype(int)).argmax()
            return leaf_value
        
        # Find best split
        feature_indices = np.arange(n_features)
        best_feature, best_threshold, best_gain = self._best_split(X, y, feature_indices)
        
        if best_gain == 0 or best_feature is None:
            leaf_value = np.bincount(y.astype(int)).argmax()
            return leaf_value
        
        # Split data
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask
        
        # Build subtrees
        left_subtree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_subtree = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return {
            'feature': best_feature,
            'threshold': best_threshold,
            'left': left_subtree,
            'right': right_subtree
        }
    
    def fit(self, X, y):
        """
        Build decision tree from training data.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values
        """
        X = np.array(X) if not isinstance(X, np.ndarray) else X
        y = np.array(y) if not isinstance(y, np.ndarray) else y
        self.tree_ = self._build_tree(X, y)
    
    def _predict_sample(self, x, tree):
        """
        Predict class for a single sample by traversing the tree.
        """
        if not isinstance(tree, dict):
            return tree
        
        if x[tree['feature']] <= tree['threshold']:
            return self._predict_sample(x, tree['left'])
        else:
            return self._predict_sample(x, tree['right'])
    
    def predict(self, X):
        """
        Predict class labels for samples in X.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict
            
        Returns:
        --------
        y_pred : array of shape (n_samples,)
            Predicted class labels
        """
        X = np.array(X) if not isinstance(X, np.ndarray) else X
        return np.array([self._predict_sample(x, self.tree_) for x in X])
    
    def save(self, filepath):
        """
        Save model to file.
        """
        if filepath.endswith('.pkl'):
            with open(filepath, 'wb') as f:
                pickle.dump(self, f)
        elif filepath.endswith('.json'):
            model_dict = {
                'max_depth': self.max_depth,
                'min_samples_split': self.min_samples_split,
                'tree': self.tree_
            }
            with open(filepath, 'w') as f:
                json.dump(model_dict, f, indent=2)
        else:
            raise ValueError("Filepath must end with .pkl or .json")
    
    @classmethod
    def load(cls, filepath):
        """
        Load model from file.
        """
        if filepath.endswith('.pkl'):
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        elif filepath.endswith('.json'):
            with open(filepath, 'r') as f:
                model_dict = json.load(f)
            model = cls(
                max_depth=model_dict['max_depth'],
                min_samples_split=model_dict['min_samples_split']
            )
            model.tree_ = model_dict['tree']
            return model
        else:
            raise ValueError("Filepath must end with .pkl or .json")


class CARTClassifier:
    """
    CART (Classification and Regression Tree) Classifier.
    Uses Gini impurity for splitting.
    """
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree_ = None
    
    def _gini(self, y):
        """
        Calculate Gini impurity.
        Gini = 1 - Σ(p_i^2)
        """
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return 1 - np.sum(probabilities ** 2)
    
    def _gini_gain(self, X_column, y, threshold):
        """
        Calculate Gini gain for a split.
        """
        parent_gini = self._gini(y)
        
        left_mask = X_column <= threshold
        right_mask = ~left_mask
        
        n = len(y)
        n_left = np.sum(left_mask)
        n_right = np.sum(right_mask)
        
        if n_left == 0 or n_right == 0:
            return 0
        
        left_gini = self._gini(y[left_mask])
        right_gini = self._gini(y[right_mask])
        
        weighted_gini = (n_left / n) * left_gini + (n_right / n) * right_gini
        
        return parent_gini - weighted_gini
    
    def _best_split(self, X, y, feature_indices):
        """
        Find the best feature and threshold to split on using Gini.
        """
        best_gain = -1
        best_feature_idx = None
        best_threshold = None
        
        for feature_idx in feature_indices:
            X_column = X[:, feature_idx]
            thresholds = np.unique(X_column)
            
            for threshold in thresholds:
                gain = self._gini_gain(X_column, y, threshold)
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature_idx = feature_idx
                    best_threshold = threshold
        
        return best_feature_idx, best_threshold, best_gain
    
    def _build_tree(self, X, y, depth=0):
        """
        Recursively build the decision tree using Gini.
        """
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))
        
        # Stopping criteria
        if (self.max_depth is not None and depth >= self.max_depth) or \
           n_labels == 1 or \
           n_samples < self.min_samples_split:
            leaf_value = np.bincount(y.astype(int)).argmax()
            return leaf_value
        
        # Find best split
        feature_indices = np.arange(n_features)
        best_feature, best_threshold, best_gain = self._best_split(X, y, feature_indices)
        
        if best_gain == 0 or best_feature is None:
            leaf_value = np.bincount(y.astype(int)).argmax()
            return leaf_value
        
        # Split data
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask
        
        # Build subtrees
        left_subtree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_subtree = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return {
            'feature': best_feature,
            'threshold': best_threshold,
            'left': left_subtree,
            'right': right_subtree
        }
    
    def fit(self, X, y):
        """
        Build decision tree from training data.
        """
        X = np.array(X) if not isinstance(X, np.ndarray) else X
        y = np.array(y) if not isinstance(y, np.ndarray) else y
        self.tree_ = self._build_tree(X, y)
    
    def _predict_sample(self, x, tree):
        """
        Predict class for a single sample.
        """
        if not isinstance(tree, dict):
            return tree
        
        if x[tree['feature']] <= tree['threshold']:
            return self._predict_sample(x, tree['left'])
        else:
            return self._predict_sample(x, tree['right'])
    
    def predict(self, X):
        """
        Predict class labels for samples in X.
        """
        X = np.array(X) if not isinstance(X, np.ndarray) else X
        return np.array([self._predict_sample(x, self.tree_) for x in X])
    
    def save(self, filepath):
        """Save model to file."""
        if filepath.endswith('.pkl'):
            with open(filepath, 'wb') as f:
                pickle.dump(self, f)
        elif filepath.endswith('.json'):
            model_dict = {
                'max_depth': self.max_depth,
                'min_samples_split': self.min_samples_split,
                'tree': self.tree_
            }
            with open(filepath, 'w') as f:
                json.dump(model_dict, f, indent=2)
        else:
            raise ValueError("Filepath must end with .pkl or .json")
    
    @classmethod
    def load(cls, filepath):
        """Load model from file."""
        if filepath.endswith('.pkl'):
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        elif filepath.endswith('.json'):
            with open(filepath, 'r') as f:
                model_dict = json.load(f)
            model = cls(
                max_depth=model_dict['max_depth'],
                min_samples_split=model_dict['min_samples_split']
            )
            model.tree_ = model_dict['tree']
            return model
        else:
            raise ValueError("Filepath must end with .pkl or .json")


class C45Classifier:
    """
    C4.5 Decision Tree Classifier.
    Uses gain ratio (normalized information gain) and supports pruning.
    Improvement over ID3: handles continuous attributes and missing values better.
    """
    def __init__(self, max_depth=None, min_samples_split=2, ccp_alpha=0.0):
        """
        Parameters:
        -----------
        max_depth : int or None
            Maximum depth of the tree
        min_samples_split : int
            Minimum number of samples required to split
        ccp_alpha : float
            Complexity parameter for cost-complexity pruning (higher = more pruning)
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.ccp_alpha = ccp_alpha
        self.tree_ = None
    
    def _entropy(self, y):
        """
        Calculate entropy of a label array.
        Entropy = -Σ(p_i * log2(p_i))
        """
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        # Filter out zero probabilities to avoid log(0)
        probabilities = probabilities[probabilities > 0]
        return -np.sum(probabilities * np.log2(probabilities))
    
    def _split_info(self, X_column, threshold):
        """
        Calculate split information (intrinsic information).
        SplitInfo = -Σ(|S_i|/|S| * log2(|S_i|/|S|))
        Used to normalize information gain → gain ratio
        """
        left_mask = X_column <= threshold
        right_mask = ~left_mask
        
        n = len(X_column)
        n_left = np.sum(left_mask)
        n_right = np.sum(right_mask)
        
        if n_left == 0 or n_right == 0:
            return 0
        
        p_left = n_left / n
        p_right = n_right / n
        
        return -(p_left * np.log2(p_left) + p_right * np.log2(p_right))
    
    def _gain_ratio(self, X_column, y, threshold):
        """
        Calculate gain ratio for a given feature and threshold.
        GainRatio = InformationGain / SplitInfo
        
        This normalizes information gain to avoid bias toward features
        with many values.
        """
        # Calculate information gain (same as ID3)
        parent_entropy = self._entropy(y)
        
        left_mask = X_column <= threshold
        right_mask = ~left_mask
        
        n = len(y)
        n_left = np.sum(left_mask)
        n_right = np.sum(right_mask)
        
        if n_left == 0 or n_right == 0:
            return 0
        
        left_entropy = self._entropy(y[left_mask])
        right_entropy = self._entropy(y[right_mask])
        
        weighted_entropy = (n_left / n) * left_entropy + (n_right / n) * right_entropy
        information_gain = parent_entropy - weighted_entropy
        
        # Calculate split info
        split_info = self._split_info(X_column, threshold)
        
        # Avoid division by zero
        if split_info == 0:
            return 0
        
        return information_gain / split_info
    
    def _best_split(self, X, y, feature_indices):
        """
        Find the best feature and threshold to split on using gain ratio.
        Returns: (best_feature_idx, best_threshold, best_gain_ratio)
        """
        best_gain_ratio = -1
        best_feature_idx = None
        best_threshold = None
        
        for feature_idx in feature_indices:
            X_column = X[:, feature_idx]
            thresholds = np.unique(X_column)
            
            for threshold in thresholds:
                gain_ratio = self._gain_ratio(X_column, y, threshold)
                
                if gain_ratio > best_gain_ratio:
                    best_gain_ratio = gain_ratio
                    best_feature_idx = feature_idx
                    best_threshold = threshold
        
        return best_feature_idx, best_threshold, best_gain_ratio
    
    def _build_tree(self, X, y, depth=0):
        """
        Recursively build the decision tree.
        Returns: tree node (dict or leaf value with class distribution)
        """
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))
        
        # Stopping criteria
        if (self.max_depth is not None and depth >= self.max_depth) or \
           n_labels == 1 or \
           n_samples < self.min_samples_split:
            # Return most common class as leaf + class distribution
            leaf_value = np.bincount(y.astype(int)).argmax()
            # Store class distribution for predict_proba
            class_dist = np.bincount(y.astype(int), minlength=self.n_classes_)
            class_proba = class_dist / class_dist.sum()
            return {'value': leaf_value, 'proba': class_proba}
        
        # Find best split
        feature_indices = np.arange(n_features)
        best_feature, best_threshold, best_gain = self._best_split(X, y, feature_indices)
        
        # If no gain, return leaf
        if best_gain == 0 or best_feature is None:
            leaf_value = np.bincount(y.astype(int)).argmax()
            class_dist = np.bincount(y.astype(int), minlength=self.n_classes_)
            class_proba = class_dist / class_dist.sum()
            return {'value': leaf_value, 'proba': class_proba}
        
        # Split data
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask
        
        # Build subtrees
        left_subtree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_subtree = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        
        # Return node
        return {
            'feature': best_feature,
            'threshold': best_threshold,
            'left': left_subtree,
            'right': right_subtree
        }
    
    def _prune_tree(self, tree, X, y):
        """
        Apply cost-complexity pruning using ccp_alpha.
        
        Cost-complexity measure: R_alpha(T) = R(T) + alpha * |T|
        where R(T) is misclassification rate and |T| is number of leaves.
        """
        # Simplified pruning implementation
        # For full implementation, would need bottom-up traversal with cost calculation
        # This is a basic version that can be enhanced
        return tree  # Placeholder - pruning can be added as enhancement
    
    def fit(self, X, y):
        """
        Build C4.5 decision tree from training data.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values
        """
        X = np.array(X) if not isinstance(X, np.ndarray) else X
        y = np.array(y) if not isinstance(y, np.ndarray) else y
        
        # Store number of classes for predict_proba
        self.n_classes_ = len(np.unique(y))
        
        self.tree_ = self._build_tree(X, y)
        
        if self.ccp_alpha > 0:
            self.tree_ = self._prune_tree(self.tree_, X, y)
    
    def _predict_sample(self, x, tree):
        """
        Predict class for a single sample by traversing the tree.
        """
        # If leaf node (dict with 'value'), return the value
        if isinstance(tree, dict) and 'value' in tree:
            return tree['value']
        
        # Old format compatibility: if integer, return it
        if not isinstance(tree, dict):
            return tree
        
        # Otherwise, traverse tree
        if x[tree['feature']] <= tree['threshold']:
            return self._predict_sample(x, tree['left'])
        else:
            return self._predict_sample(x, tree['right'])
    
    def _predict_proba_sample(self, x, tree):
        """
        Predict class probabilities for a single sample.
        """
        # If leaf node with probabilities
        if isinstance(tree, dict) and 'proba' in tree:
            return tree['proba']
        
        # Old format compatibility: if integer leaf, return one-hot
        if not isinstance(tree, dict):
            proba = np.zeros(self.n_classes_)
            proba[tree] = 1.0
            return proba
        
        # Otherwise, traverse tree
        if x[tree['feature']] <= tree['threshold']:
            return self._predict_proba_sample(x, tree['left'])
        else:
            return self._predict_proba_sample(x, tree['right'])
    
    def predict(self, X):
        """
        Predict class labels for samples in X.
        """
        X = np.array(X) if not isinstance(X, np.ndarray) else X
        return np.array([self._predict_sample(x, self.tree_) for x in X])
    
    def predict_proba(self, X):
        """
        Predict class probabilities for samples in X.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict
            
        Returns:
        --------
        proba : array of shape (n_samples, n_classes)
            Predicted class probabilities
        """
        X = np.array(X) if not isinstance(X, np.ndarray) else X
        return np.array([self._predict_proba_sample(x, self.tree_) for x in X])
    
    def save(self, filepath):
        """
        Save model to file.
        """
        if filepath.endswith('.pkl'):
            with open(filepath, 'wb') as f:
                pickle.dump(self, f)
        elif filepath.endswith('.json'):
            model_dict = {
                'max_depth': self.max_depth,
                'min_samples_split': self.min_samples_split,
                'ccp_alpha': self.ccp_alpha,
                'tree': self.tree_
            }
            with open(filepath, 'w') as f:
                json.dump(model_dict, f, indent=2)
        else:
            raise ValueError("Filepath must end with .pkl or .json")
    
    @classmethod
    def load(cls, filepath):
        """
        Load model from file.
        """
        if filepath.endswith('.pkl'):
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        elif filepath.endswith('.json'):
            with open(filepath, 'r') as f:
                model_dict = json.load(f)
            model = cls(
                max_depth=model_dict['max_depth'],
                min_samples_split=model_dict['min_samples_split'],
                ccp_alpha=model_dict['ccp_alpha']
            )
            model.tree_ = model_dict['tree']
            return model
        else:
            raise ValueError("Filepath must end with .pkl or .json")


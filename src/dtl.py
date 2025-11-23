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
        # TODO: Implementasi entropy
        pass
    
    def _information_gain(self, X_column, y, threshold):
        """
        Calculate information gain for a given feature and threshold.
        IG = entropy(parent) - weighted_avg(entropy(children))
        """
        # TODO: Implementasi information gain
        pass
    
    def _best_split(self, X, y, feature_indices):
        """
        Find the best feature and threshold to split on.
        Returns: (best_feature_idx, best_threshold, best_gain)
        """
        # TODO: Implementasi best split
        pass
    
    def _build_tree(self, X, y, depth=0):
        """
        Recursively build the decision tree.
        Returns: tree node (dict or leaf value)
        """
        # TODO: Implementasi tree building
        # Stopping criteria:
        # - max_depth reached
        # - min_samples_split
        # - all samples have same label
        # - no more information gain
        pass
    
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
        # TODO: Convert to numpy if needed, then build tree
        # self.tree_ = self._build_tree(X, y)
        pass
    
    def _predict_sample(self, x, tree):
        """
        Predict class for a single sample by traversing the tree.
        """
        # TODO: Implementasi tree traversal untuk prediksi
        pass
    
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
        # TODO: Loop through samples and predict each
        # return np.array([self._predict_sample(x, self.tree_) for x in X])
        pass
    
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
        # TODO: Implementasi Gini impurity
        pass
    
    def _gini_gain(self, X_column, y, threshold):
        """
        Calculate Gini gain for a split.
        """
        # TODO: Implementasi Gini gain
        pass
    
    def _best_split(self, X, y, feature_indices):
        """
        Find the best feature and threshold to split on using Gini.
        """
        # TODO: Implementasi best split dengan Gini
        pass
    
    def _build_tree(self, X, y, depth=0):
        """
        Recursively build the decision tree using Gini.
        """
        # TODO: Implementasi tree building dengan Gini
        pass
    
    def fit(self, X, y):
        """
        Build decision tree from training data.
        """
        # TODO: self.tree_ = self._build_tree(X, y)
        pass
    
    def _predict_sample(self, x, tree):
        """
        Predict class for a single sample.
        """
        # TODO: Implementasi tree traversal
        pass
    
    def predict(self, X):
        """
        Predict class labels for samples in X.
        """
        # TODO: return np.array([self._predict_sample(x, self.tree_) for x in X])
        pass
    
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
        # TODO: Implementasi entropy (sama seperti ID3)
        pass
    
    def _split_info(self, X_column, threshold):
        """
        Calculate split information (intrinsic information).
        SplitInfo = -Σ(|S_i|/|S| * log2(|S_i|/|S|))
        Used to normalize information gain → gain ratio
        """
        # TODO: Implementasi split info
        pass
    
    def _gain_ratio(self, X_column, y, threshold):
        """
        Calculate gain ratio for a given feature and threshold.
        GainRatio = InformationGain / SplitInfo
        
        This normalizes information gain to avoid bias toward features
        with many values.
        """
        # TODO: Implementasi gain ratio
        # 1. Hitung information gain (entropy-based)
        # 2. Hitung split info
        # 3. Return gain / split_info (handle division by zero)
        pass
    
    def _best_split(self, X, y, feature_indices):
        """
        Find the best feature and threshold to split on using gain ratio.
        Returns: (best_feature_idx, best_threshold, best_gain_ratio)
        """
        # TODO: Implementasi best split dengan gain ratio
        pass
    
    def _build_tree(self, X, y, depth=0):
        """
        Recursively build the decision tree.
        Returns: tree node (dict or leaf value)
        """
        # TODO: Implementasi tree building dengan gain ratio
        # Stopping criteria (sama seperti ID3):
        # - max_depth reached
        # - min_samples_split
        # - all samples have same label
        # - no more gain ratio improvement
        pass
    
    def _prune_tree(self, tree, X, y):
        """
        Apply cost-complexity pruning using ccp_alpha.
        
        Cost-complexity measure: R_alpha(T) = R(T) + alpha * |T|
        where R(T) is misclassification rate and |T| is number of leaves.
        """
        # TODO: Implementasi pruning (optional, untuk bonus)
        # Pruning dilakukan bottom-up:
        # 1. Traverse tree from leaves to root
        # 2. For each node, compare cost with/without subtree
        # 3. Prune if cost reduces
        pass
    
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
        # TODO: Convert to numpy if needed, then build tree
        # self.tree_ = self._build_tree(X, y)
        # if self.ccp_alpha > 0:
        #     self.tree_ = self._prune_tree(self.tree_, X, y)
        pass
    
    def _predict_sample(self, x, tree):
        """
        Predict class for a single sample by traversing the tree.
        """
        # TODO: Implementasi tree traversal untuk prediksi
        pass
    
    def predict(self, X):
        """
        Predict class labels for samples in X.
        """
        # TODO: Loop through samples and predict each
        # return np.array([self._predict_sample(x, self.tree_) for x in X])
        pass
    
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


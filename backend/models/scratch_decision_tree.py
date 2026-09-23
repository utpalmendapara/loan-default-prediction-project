"""
scratch_decision_tree.py
========================
Custom Decision Tree Classifier implemented from scratch using only Python and NumPy.
No scikit-learn or machine learning libraries are used in the core tree building,
splitting criteria, or prediction logic.

SOP Week 3 Milestone: "Implement selected Algorithm without use of Library."

Key Concepts Implemented:
1. Gini Impurity: Measurement of node impurity.
   Gini = 1 - sum(p_i^2)
2. Best Split Finding: Evaluates all candidate feature thresholds to maximize Information Gain (Gini reduction).
3. Recursive Tree Construction: Recursively partitions data until stopping criteria (max_depth, min_samples_split, pure node).
4. Inference: Recursively traverses tree nodes to produce predicted class and class probabilities.
"""

from typing import Optional, Union, Tuple, Dict, Any
import numpy as np


class DecisionTreeNode:
    """
    Represents a single node in the Decision Tree.
    """
    def __init__(
        self,
        feature_idx: Optional[int] = None,
        threshold: Optional[float] = None,
        left: Optional["DecisionTreeNode"] = None,
        right: Optional["DecisionTreeNode"] = None,
        *,
        value: Optional[int] = None,
        probabilities: Optional[np.ndarray] = None
    ):
        # Decision node attributes
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right

        # Leaf node attributes
        self.value = value
        self.probabilities = probabilities

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


class ScratchDecisionTreeClassifier:
    """
    Binary / Multi-class Decision Tree Classifier built from scratch with NumPy.
    """
    def __init__(
        self,
        max_depth: int = 6,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Optional[Union[int, float, str]] = None,
        random_state: Optional[int] = 42
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.root: Optional[DecisionTreeNode] = None
        self.n_classes_: int = 0
        self.classes_: Optional[np.ndarray] = None
        self.n_features_: int = 0
        self.feature_importances_: Optional[np.ndarray] = None

    def _gini(self, y: np.ndarray) -> float:
        """
        Calculates Gini Impurity for a label array y:
        Gini = 1 - sum(p_k^2)
        """
        n = len(y)
        if n == 0:
            return 0.0
        counts = np.bincount(y, minlength=self.n_classes_)
        probabilities = counts / n
        return 1.0 - float(np.sum(probabilities ** 2))

    def _best_split(
        self, X: np.ndarray, y: np.ndarray, feat_indices: np.ndarray
    ) -> Tuple[Optional[int], Optional[float], float]:
        """
        Finds the feature and threshold that maximize Gini information gain.
        """
        best_gain = -1.0
        best_feat = None
        best_thresh = None

        n_samples = len(y)
        parent_gini = self._gini(y)
        if parent_gini == 0.0:
            return None, None, 0.0

        for feat_idx in feat_indices:
            values = X[:, feat_idx]
            unique_vals = np.unique(values)
            if len(unique_vals) <= 1:
                continue

            # Evaluate percentiles or midpoints for fast, robust splits
            if len(unique_vals) > 20:
                thresholds = np.percentile(unique_vals, np.linspace(5, 95, 15))
            else:
                thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2.0

            for thresh in thresholds:
                left_mask = values <= thresh
                right_mask = ~left_mask

                n_left = np.sum(left_mask)
                n_right = n_samples - n_left

                if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                    continue

                gini_left = self._gini(y[left_mask])
                gini_right = self._gini(y[right_mask])

                # Weighted child impurity
                child_gini = (n_left / n_samples) * gini_left + (n_right / n_samples) * gini_right
                gain = parent_gini - child_gini

                if gain > best_gain:
                    best_gain = gain
                    best_feat = feat_idx
                    best_thresh = thresh

        return best_feat, best_thresh, (best_gain if best_gain > 0 else 0.0)

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> DecisionTreeNode:
        """
        Recursively builds the decision tree.
        """
        n_samples, n_features = X.shape
        counts = np.bincount(y, minlength=self.n_classes_)
        probabilities = counts / (n_samples if n_samples > 0 else 1)
        majority_class = int(np.argmax(counts))

        # Check stopping criteria
        if (
            depth >= self.max_depth
            or n_samples < self.min_samples_split
            or len(np.unique(y)) <= 1
        ):
            return DecisionTreeNode(value=majority_class, probabilities=probabilities)

        # Select features to evaluate
        rng = np.random.RandomState(self.random_state + depth if self.random_state else None)
        feat_indices = np.arange(n_features)
        if isinstance(self.max_features, int) and self.max_features < n_features:
            feat_indices = rng.choice(feat_indices, size=self.max_features, replace=False)
        elif isinstance(self.max_features, float) and 0 < self.max_features < 1:
            size = max(1, int(self.max_features * n_features))
            feat_indices = rng.choice(feat_indices, size=size, replace=False)

        best_feat, best_thresh, gain = self._best_split(X, y, feat_indices)

        if best_feat is None or gain <= 1e-7:
            return DecisionTreeNode(value=majority_class, probabilities=probabilities)

        # Accumulate feature importance (weighted information gain)
        if self.feature_importances_ is not None:
            self.feature_importances_[best_feat] += gain * (n_samples / len(self._total_y))

        # Split and recurse
        left_mask = X[:, best_feat] <= best_thresh
        right_mask = ~left_mask

        left_node = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_node = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return DecisionTreeNode(
            feature_idx=best_feat,
            threshold=best_thresh,
            left=left_node,
            right=right_node,
            value=majority_class,
            probabilities=probabilities
        )

    def fit(self, X: Union[np.ndarray, Any], y: Union[np.ndarray, Any]) -> "ScratchDecisionTreeClassifier":
        """
        Fit the decision tree classifier to training dataset.
        """
        if hasattr(X, "to_numpy"):
            X = X.to_numpy()
        if hasattr(y, "to_numpy"):
            y = y.to_numpy()

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)

        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        self.n_features_ = X.shape[1]
        self._total_y = y
        self.feature_importances_ = np.zeros(self.n_features_, dtype=np.float64)

        self.root = self._build_tree(X, y, depth=0)

        # Normalize feature importances so they sum to 1.0
        total_imp = np.sum(self.feature_importances_)
        if total_imp > 0:
            self.feature_importances_ /= total_imp

        return self

    def _traverse_row(self, row: np.ndarray, node: DecisionTreeNode) -> DecisionTreeNode:
        if node.is_leaf or node.feature_idx is None:
            return node
        if row[node.feature_idx] <= node.threshold:
            return self._traverse_row(row, node.left)
        return self._traverse_row(row, node.right)

    def predict_proba(self, X: Union[np.ndarray, Any]) -> np.ndarray:
        """
        Returns class probabilities for test samples.
        """
        if hasattr(X, "to_numpy"):
            X = X.to_numpy()
        X = np.asarray(X, dtype=np.float64)

        probabilities = []
        for row in X:
            node = self._traverse_row(row, self.root)
            probabilities.append(node.probabilities)
        return np.array(probabilities)

    def predict(self, X: Union[np.ndarray, Any]) -> np.ndarray:
        """
        Predicts binary / multi-class labels for test samples.
        """
        prob = self.predict_proba(X)
        return np.argmax(prob, axis=1)

    def score(self, X: Union[np.ndarray, Any], y: Union[np.ndarray, Any]) -> float:
        """
        Computes accuracy score on test set.
        """
        if hasattr(y, "to_numpy"):
            y = y.to_numpy()
        preds = self.predict(X)
        return float(np.mean(preds == y))

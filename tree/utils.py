# # # """
# # # You can add your own functions here according to your decision tree implementation.
# # # There is no restriction on following the below template, these fucntions are here to simply help you.
# # # """

# # # import pandas as pd

# # # def one_hot_encoding(X: pd.DataFrame) -> pd.DataFrame:
# # #     """
# # #     Function to perform one hot encoding on the input data
# # #     """

# # #     pass

# # # def check_ifreal(y: pd.Series) -> bool:
# # #     """
# # #     Function to check if the given series has real or discrete values
# # #     """

# # #     pass


# # # def entropy(Y: pd.Series) -> float:
# # #     """
# # #     Function to calculate the entropy
# # #     """

# # #     pass


# # # def gini_index(Y: pd.Series) -> float:
# # #     """
# # #     Function to calculate the gini index
# # #     """

# # #     pass


# # # def information_gain(Y: pd.Series, attr: pd.Series, criterion: str) -> float:
# # #     """
# # #     Function to calculate the information gain using criterion (entropy, gini index or MSE)
# # #     """

# # #     pass


# # # def opt_split_attribute(X: pd.DataFrame, y: pd.Series, criterion, features: pd.Series):
# # #     """
# # #     Function to find the optimal attribute to split about.
# # #     If needed you can split this function into 2, one for discrete and one for real valued features.
# # #     You can also change the parameters of this function according to your implementation.

# # #     features: pd.Series is a list of all the attributes we have to split upon

# # #     return: attribute to split upon
# # #     """

# # #     # According to wheather the features are real or discrete valued and the criterion, find the attribute from the features series with the maximum information gain (entropy or varinace based on the type of output) or minimum gini index (discrete output).

# # #     pass


# # # def split_data(X: pd.DataFrame, y: pd.Series, attribute, value):
# # #     """
# # #     Funtion to split the data according to an attribute.
# # #     If needed you can split this function into 2, one for discrete and one for real valued features.
# # #     You can also change the parameters of this function according to your implementation.

# # #     attribute: attribute/feature to split upon
# # #     value: value of that attribute to split upon

# # #     return: splitted data(Input and output)
# # #     """

# # #     # Split the data based on a particular value of a particular attribute. You may use masking as a tool to split the data.

# # #     pass



# # """
# # Utility functions for the Decision Tree implementation.
# # Only numpy/pandas/matplotlib allowed. sklearn is used elsewhere only for experiments.
# # """

# # from __future__ import annotations
# # import numpy as np
# # import pandas as pd


# # def one_hot_encoding(X: pd.DataFrame) -> pd.DataFrame:
# #     """
# #     One-hot encode any categorical/object/boolean columns to real-valued 0/1 columns.
# #     Leaves numeric (int/float) columns as-is.
# #     """
# #     if X.empty:
# #         return X.copy()

# #     # Identify categorical-like columns
# #     cat_cols = [
# #         c for c in X.columns
# #         if pd.api.types.is_categorical_dtype(X[c])
# #         or pd.api.types.is_object_dtype(X[c])
# #         or pd.api.types.is_bool_dtype(X[c])
# #         or (pd.api.types.is_integer_dtype(X[c]) and X[c].nunique() <= max(10, int(0.05 * len(X))))
# #     ]

# #     if len(cat_cols) == 0:
# #         return X.astype(float)

# #     # Use get_dummies for robust one-hot. Drop first=False so each category has its own indicator.
# #     X_encoded = pd.get_dummies(X, columns=cat_cols, drop_first=False)
# #     # Ensure float dtype
# #     return X_encoded.astype(float)


# # def check_ifreal(y: pd.Series) -> bool:
# #     """
# #     Returns True if y should be treated as a real-valued target (regression).
# #     We treat floats as real; categories/objects/bools/integers as discrete.
# #     """
# #     if y.empty:
# #         return False
# #     if pd.api.types.is_float_dtype(y):
# #         return True
# #     if pd.api.types.is_categorical_dtype(y) or pd.api.types.is_object_dtype(y) or pd.api.types.is_bool_dtype(y):
# #         return False
# #     if pd.api.types.is_integer_dtype(y):
# #         return False
# #     # Fallback: if unique proportion is very high, treat as real
# #     return y.nunique() > max(20, 0.2 * len(y))


# # def entropy(Y: pd.Series) -> float:
# #     """
# #     Shannon entropy for discrete Y.
# #     """
# #     if len(Y) == 0:
# #         return 0.0
# #     counts = Y.value_counts().values.astype(float)
# #     probs = counts / counts.sum()
# #     # Avoid log(0)
# #     probs = probs[probs > 0]
# #     return float(-(probs * np.log2(probs)).sum()) if probs.size else 0.0


# # def gini_index(Y: pd.Series) -> float:
# #     """
# #     Gini impurity for discrete Y.
# #     """
# #     if len(Y) == 0:
# #         return 0.0
# #     counts = Y.value_counts().values.astype(float)
# #     probs = counts / counts.sum()
# #     return float(1.0 - np.sum(probs ** 2))


# # def mse(Y: pd.Series) -> float:
# #     """
# #     Mean squared error around the mean (node variance times n/n == variance).
# #     """
# #     if len(Y) == 0:
# #         return 0.0
# #     m = Y.mean()
# #     return float(np.mean((Y - m) ** 2))


# # def information_gain(Y: pd.Series, attr: pd.Series, criterion: str) -> float:
# #     """
# #     Information gain of splitting Y by a *discrete* attribute 'attr'.
# #     - If criterion in {"information_gain", "entropy"} => use entropy reduction.
# #     - If criterion == "gini_index" => use gini reduction (i.e., impurity decrease).
# #     - If criterion == "mse" => use MSE reduction (for regression) across groups of attr values.

# #     Note: This function is used when splitting on categorical attributes directly.
# #     In our tree we one-hot encode inputs, so we mostly use threshold-based splits,
# #     but this is provided for completeness and reuse.
# #     """
# #     assert len(Y) == len(attr)
# #     if len(Y) == 0:
# #         return 0.0

# #     # Partition by attribute values
# #     groups = []
# #     for v, idx in attr.groupby(attr).groups.items():
# #         groups.append(Y.loc[idx])

# #     if criterion in ("information_gain", "entropy"):
# #         base = entropy(Y)
# #         weighted = sum((len(g) / len(Y)) * entropy(g) for g in groups)
# #         return float(base - weighted)
# #     elif criterion == "gini_index":
# #         base = gini_index(Y)
# #         weighted = sum((len(g) / len(Y)) * gini_index(g) for g in groups)
# #         # We return impurity decrease (larger is better)
# #         return float(base - weighted)
# #     elif criterion == "mse":
# #         base = mse(Y)
# #         weighted = sum((len(g) / len(Y)) * mse(g) for g in groups)
# #         return float(base - weighted)
# #     else:
# #         raise ValueError(f"Unknown criterion: {criterion}")


# # def _weighted_impurity(y_left: pd.Series, y_right: pd.Series, task: str, criterion: str) -> float:
# #     """
# #     Helper: returns weighted impurity for a proposed binary split.
# #     Smaller is better *if* you are minimizing impurity directly,
# #     but we will compute information gain as base - weighted later.
# #     """
# #     n = len(y_left) + len(y_right)
# #     if n == 0:
# #         return 0.0

# #     wl = len(y_left) / n
# #     wr = len(y_right) / n

# #     if task == "classification":
# #         if criterion == "gini_index":
# #             return wl * gini_index(y_left) + wr * gini_index(y_right)
# #         else:  # entropy
# #             return wl * entropy(y_left) + wr * entropy(y_right)
# #     else:  # regression
# #         return wl * mse(y_left) + wr * mse(y_right)


# # def opt_split_attribute(
# #     X: pd.DataFrame, y: pd.Series, criterion: str, features: pd.Series | list | None
# # ):
# #     """
# #     Find the best (feature, threshold) binary split that maximizes information gain.
# #     - For one-hot or binary columns, threshold 0.5 is sufficient.
# #     - For real-valued columns, consider thresholds at midpoints between sorted unique values.

# #     Returns:
# #         best_feature (str) or None,
# #         best_threshold (float) or None,
# #         best_gain (float)
# #     """
# #     if features is None:
# #         feat_list = list(X.columns)
# #     else:
# #         feat_list = list(features)

# #     task = "regression" if check_ifreal(y) else "classification"

# #     # Base impurity
# #     if task == "classification":
# #         base = entropy(y) if criterion == "information_gain" else gini_index(y)
# #     else:
# #         base = mse(y)

# #     best_gain = -np.inf
# #     best_feature = None
# #     best_threshold = None

# #     for f in feat_list:
# #         x = X[f].values
# #         # Skip constant features
# #         if np.all(x == x[0]):
# #             continue

# #         # Candidate thresholds
# #         unique_vals = np.unique(x)
# #         if unique_vals.size == 2 and set(unique_vals) <= {0.0, 1.0}:
# #             thresholds = [0.5]
# #         else:
# #             # Sort by feature and only test midpoints where target changes
# #             order = np.argsort(x)
# #             x_sorted = x[order]
# #             y_sorted = y.values[order]
# #             # To avoid O(N^2), only consider boundaries where x changes
# #             boundaries = np.where(np.diff(x_sorted) != 0)[0]
# #             thresholds = []
# #             for b in boundaries:
# #                 t = (x_sorted[b] + x_sorted[b + 1]) / 2.0
# #                 thresholds.append(float(t))

# #             if len(thresholds) == 0:
# #                 continue

# #         # Evaluate thresholds
# #         for thr in thresholds:
# #             left_idx = X[f] <= thr
# #             y_left = y[left_idx]
# #             y_right = y[~left_idx]
# #             if len(y_left) == 0 or len(y_right) == 0:
# #                 continue
# #             weighted_imp = _weighted_impurity(y_left, y_right, task, criterion)
# #             gain = base - weighted_imp
# #             if gain > best_gain:
# #                 best_gain = gain
# #                 best_feature = f
# #                 best_threshold = thr

# #     if best_gain == -np.inf:
# #         # No split improved impurity
# #         return None, None, 0.0
# #     return best_feature, best_threshold, float(best_gain)


# # def split_data(X: pd.DataFrame, y: pd.Series, attribute: str, threshold: float):
# #     """
# #     Binary split: left = X[attribute] <= threshold, right = > threshold.
# #     Returns (X_left, y_left, X_right, y_right)
# #     """
# #     mask = X[attribute] <= threshold
# #     return X.loc[mask], y.loc[mask], X.loc[~mask], y.loc[~mask]

# # # from graphviz import Digraph

# # # def export_tree_png(root, filename="tree.png"):
# # #     dot = Digraph()
    
# # #     def add_nodes(node, node_id=0):
# # #         if node is None:
# # #             return node_id

# # #         if hasattr(node, "prediction") and node.prediction is not None:
# # #             # Leaf node
# # #             dot.node(str(node_id), f"Leaf\nvalue={node.prediction}\nn={node.n_samples}", shape="box", style="filled", color="lightgrey")
# # #             return node_id
# # #         else:
# # #             # Decision node
# # #             dot.node(str(node_id), f"X[{node.feature}] <= {node.threshold:.4f}\nn={node.n_samples}")
            
# # #             left_id = node_id + 1
# # #             left_id = add_nodes(node.left, left_id)
# # #             dot.edge(str(node_id), str(node_id+1), label="Yes")

# # #             right_id = left_id + 1
# # #             right_id = add_nodes(node.right, right_id)
# # #             dot.edge(str(node_id), str(right_id), label="No")

# # #             return right_id

# # #     add_nodes(root)
# # #     dot.render(filename, format="png", cleanup=True)


# # from graphviz import Digraph

# # def export_tree_png(root, filename="tree", metrics=None):
# #     """
# #     Exports a decision tree to PNG with optional performance metrics.
    
# #     Args:
# #         root: root node of the decision tree
# #         filename (str): output filename (without extension)
# #         metrics (dict): performance metrics to display in the image
# #                         e.g. {"RMSE": 0.36, "MAE": 0.24}
# #                         or   {"Accuracy": 0.9, "Precision": [..], "Recall": [..]}
# #     """
# #     dot = Digraph(comment="Decision Tree")

# #     def add_nodes(node, node_id=0):
# #         if node is None:
# #             return node_id

# #         if hasattr(node, "prediction") and node.prediction is not None:
# #             # Leaf node
# #             dot.node(
# #                 str(node_id),
# #                 f"Leaf\nn={node.n_samples}\nvalue={node.prediction}",
# #                 shape="box",
# #                 style="filled",
# #                 color="lightgrey"
# #             )
# #             return node_id
# #         else:
# #             # Internal decision node
# #             dot.node(
# #                 str(node_id),
# #                 f"X[{node.feature}] <= {node.threshold:.4f}\n n={node.n_samples}"
# #             )
# #             left_id = node_id + 1
# #             left_id = add_nodes(node.left, left_id)
# #             dot.edge(str(node_id), str(node_id+1), label="Yes")

# #             right_id = left_id + 1
# #             right_id = add_nodes(node.right, right_id)
# #             dot.edge(str(node_id), str(right_id), label="No")

# #             return right_id

# #     # Draw tree
# #     add_nodes(root)

# #     # Add metrics box (as a separate node)
# #     if metrics:
# #         metrics_text = "\n".join([f"{k}: {v}" for k, v in metrics.items()])
# #         dot.node("metrics", metrics_text, shape="note", color="blue")

# #     # Render PNG
# #     dot.render(filename, format="png", cleanup=True)





# # tree/utils.py
# """
# Utility functions for the decision tree assignment.
# """
# import pandas as pd
# import numpy as np
# from pandas.api import types as pdtypes
# from typing import Tuple, Optional


# def one_hot_encoding(X: pd.DataFrame) -> pd.DataFrame:
#     """
#     One-hot encode categorical columns. Return DataFrame with only numeric columns.
#     Uses pandas.get_dummies which preserves numeric columns as-is.
#     """
#     if not isinstance(X, pd.DataFrame):
#         raise ValueError("X must be a pandas DataFrame")
#     # Convert categorical typed columns or object columns to dummies.
#     return pd.get_dummies(X, drop_first=False)


# def check_ifreal(y: pd.Series) -> bool:
#     """
#     Return True if y should be treated as real-valued (regression).
#     We'll treat y as categorical/ discrete if it is an actual pandas categorical dtype.
#     If y is numeric but categorical dtype is used, it's discrete.
#     """
#     if pdtypes.is_categorical_dtype(y):
#         return False
#     # If dtype is object/str -> discrete (categorical)
#     if pdtypes.is_object_dtype(y):
#         return False
#     # numeric dtypes (int/float) -> treat as real unless it's categorical dtype
#     if pdtypes.is_numeric_dtype(y):
#         # Many datasets use int labels for classification but the student test uses dtype="category".
#         # We'll treat numeric as real unless categorical dtype.
#         return True
#     # Fallback: treat as discrete
#     return False


# def entropy(Y: pd.Series) -> float:
#     """
#     Entropy of labels Y (classification).
#     Y may contain values of any hashable type.
#     """
#     if Y.size == 0:
#         return 0.0
#     counts = Y.value_counts(normalize=True)
#     probs = counts.values
#     # avoid log2(0)
#     probs = probs[probs > 0]
#     return -np.sum(probs * np.log2(probs))


# def gini_index(Y: pd.Series) -> float:
#     """
#     Gini index for labels Y (classification).
#     """
#     if Y.size == 0:
#         return 0.0
#     probs = Y.value_counts(normalize=True).values
#     return 1.0 - np.sum(probs ** 2)


# def mse(Y: pd.Series) -> float:
#     """
#     Mean squared error (variance) of Y (regression). We use variance (unbiased estimator not needed).
#     """
#     if Y.size == 0:
#         return 0.0
#     return float(np.mean((Y - Y.mean()) ** 2))


# def opt_split_attribute(
#     X: pd.DataFrame,
#     y: pd.Series,
#     criterion: str,
#     features
# ) -> Tuple[Optional[str], Optional[float], float]:
#     """
#     Find the best attribute and threshold to split upon for binary splits.
#     Returns (best_feature_name, best_threshold, best_score_gain)
#     - For numeric features we search possible thresholds (midpoints between sorted unique values).
#     - We assume X contains only numeric columns (one-hot encoding applied previously if necessary).
#     """
#     # Determine if target is regression
#     is_regression = check_ifreal(y)

#     # Define impurity function for classification/regression
#     def impurity(Y):
#         if is_regression:
#             return mse(Y)
#         else:
#             if criterion == "information_gain":
#                 return entropy(Y)
#             elif criterion == "gini_index":
#                 return gini_index(Y)
#             else:
#                 # default to entropy
#                 return entropy(Y)

#     parent_impurity = impurity(y)
#     best_gain = -np.inf
#     best_feature = None
#     best_threshold = None

#     n = y.size
#     if n == 0:
#         return None, None, 0.0

#     for feature in features:
#         col = X[feature]
#         # If column is constant skip
#         if col.nunique() <= 1:
#             continue

#         # Candidate thresholds: midpoints between sorted unique values
#         uniq = np.sort(col.unique())
#         # If only two values, midpoint is fine
#         thresholds = []
#         if uniq.size == 1:
#             thresholds = [uniq[0]]
#         else:
#             thresholds = (uniq[:-1] + uniq[1:]) / 2.0

#         # Evaluate each threshold
#         for t in thresholds:
#             left_mask = col <= t
#             right_mask = ~left_mask
#             left_y = y[left_mask]
#             right_y = y[right_mask]

#             # skip empty splits
#             if left_y.size == 0 or right_y.size == 0:
#                 continue

#             # Weighted impurity
#             w_impurity = (left_y.size / n) * impurity(left_y) + (right_y.size / n) * impurity(right_y)

#             # For classification using "information_gain" or "gini_index", we consider gain = parent - weighted
#             # For regression (MSE) we also consider gain = parent - weighted (reduce MSE)
#             gain = parent_impurity - w_impurity

#             # Choose max gain
#             if gain > best_gain:
#                 best_gain = gain
#                 best_feature = feature
#                 best_threshold = float(t)

#     if best_gain == -np.inf:
#         return None, None, 0.0
#     return best_feature, best_threshold, float(best_gain)


# def split_data(X: pd.DataFrame, y: pd.Series, attribute: str, value: float):
#     """
#     Binary split on a numeric attribute at threshold value.
#     Returns (X_left, y_left, X_right, y_right)
#     Left: attribute <= value
#     Right: attribute > value
#     """
#     mask_left = X[attribute] <= value
#     X_left = X[mask_left]
#     y_left = y[mask_left]
#     X_right = X[~mask_left]
#     y_right = y[~mask_left]
#     return X_left, y_left, X_right, y_right



# tree/utils.py
"""
Utility functions for the decision tree assignment.
"""
import pandas as pd
import numpy as np
from pandas.api import types as pdtypes
from typing import Tuple, Optional


def one_hot_encoding(X: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode categorical columns and return a numeric DataFrame.
    Numeric columns are kept as-is.
    """
    if not isinstance(X, pd.DataFrame):
        raise ValueError("X must be a pandas DataFrame")
    # Use pandas.get_dummies for simplicity (keeps numeric as-is)
    return pd.get_dummies(X, drop_first=False)


def check_ifreal(y: pd.Series) -> bool:
    """
    Return True if target y should be treated as real-valued (regression).
    We treat y as categorical (classification) if it is pandas Categorical dtype.
    If dtype is object -> categorical. Numeric dtype -> regression (unless categorical).
    """
    if pdtypes.is_categorical_dtype(y):
        return False
    if pdtypes.is_object_dtype(y):
        return False
    if pdtypes.is_numeric_dtype(y):
        # tests in usage.py pass dtype="category" for classification,
        # so treat numeric as regression here.
        return True
    return False


def entropy(Y: pd.Series) -> float:
    """
    Compute entropy of label distribution in Y.
    """
    if Y.size == 0:
        return 0.0
    probs = Y.value_counts(normalize=True).values
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def gini_index(Y: pd.Series) -> float:
    """
    Compute Gini index of label distribution in Y.
    """
    if Y.size == 0:
        return 0.0
    probs = Y.value_counts(normalize=True).values
    return float(1.0 - np.sum(probs ** 2))


def mse(Y: pd.Series) -> float:
    """
    Mean squared error (variance) of Y.
    """
    if Y.size == 0:
        return 0.0
    return float(np.mean((Y - Y.mean()) ** 2))


def opt_split_attribute(
    X: pd.DataFrame,
    y: pd.Series,
    criterion: str,
    features
) -> Tuple[Optional[str], Optional[float], float]:
    """
    Find best feature and threshold to split upon.
    Returns (best_feature, best_threshold, best_gain).
    Assumes X is numeric (one-hot encoded if originally categorical).
    Uses:
      - entropy (information_gain) or gini_index for classification
      - mse (variance) for regression
    """
    is_regression = check_ifreal(y)

    def impurity(Y):
        if is_regression:
            return mse(Y)
        else:
            if criterion == "information_gain":
                return entropy(Y)
            else:
                return gini_index(Y)

    parent_imp = impurity(y)
    best_gain = -np.inf
    best_feature = None
    best_threshold = None
    n = len(y)
    if n == 0:
        return None, None, 0.0

    for feat in features:
        col = X[feat]
        if col.nunique() <= 1:
            continue
        uniq = np.sort(col.unique())
        # candidate thresholds: midpoints between unique sorted values
        if uniq.size == 1:
            thresholds = [uniq[0]]
        else:
            thresholds = (uniq[:-1] + uniq[1:]) / 2.0

        for t in thresholds:
            left_mask = col <= t
            right_mask = ~left_mask
            left_y = y[left_mask]
            right_y = y[right_mask]
            if left_y.size == 0 or right_y.size == 0:
                continue
            w_imp = (left_y.size / n) * impurity(left_y) + (right_y.size / n) * impurity(right_y)
            gain = parent_imp - w_imp
            if gain > best_gain:
                best_gain = gain
                best_feature = feat
                best_threshold = float(t)

    if best_gain == -np.inf:
        return None, None, 0.0
    return best_feature, best_threshold, float(best_gain)


def split_data(X: pd.DataFrame, y: pd.Series, attribute, value):
    """
    Binary split of X,y on attribute at threshold value:
      left: attribute <= value
      right: attribute > value
    Returns (X_left, y_left, X_right, y_right)
    """
    mask_left = X[attribute] <= value
    X_left = X[mask_left].copy()
    y_left = y[mask_left].copy()
    X_right = X[~mask_left].copy()
    y_right = y[~mask_left].copy()
    return X_left, y_left, X_right, y_right

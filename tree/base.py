# # # # # """
# # # # # The current code given is for the Assignment 1.
# # # # # You will be expected to use this to make trees for:
# # # # # > discrete input, discrete output
# # # # # > real input, real output
# # # # # > real input, discrete output
# # # # # > discrete input, real output
# # # # # """
# # # # # from dataclasses import dataclass
# # # # # from typing import Literal

# # # # # import numpy as np
# # # # # import pandas as pd
# # # # # import matplotlib.pyplot as plt
# # # # # from tree.utils import *

# # # # # np.random.seed(42)


# # # # # @dataclass
# # # # # class DecisionTree:
# # # # #     criterion: Literal["information_gain", "gini_index"]  # criterion won't be used for regression
# # # # #     max_depth: int  # The maximum depth the tree can grow to

# # # # #     def __init__(self, criterion, max_depth=5):
# # # # #         self.criterion = criterion
# # # # #         self.max_depth = max_depth

# # # # #     def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
# # # # #         """
# # # # #         Function to train and construct the decision tree
# # # # #         """

# # # # #         # If you wish your code can have cases for different types of input and output data (discrete, real)
# # # # #         # Use the functions from utils.py to find the optimal attribute to split upon and then construct the tree accordingly.
# # # # #         # You may(according to your implemetation) need to call functions recursively to construct the tree. 

# # # # #         pass

# # # # #     def predict(self, X: pd.DataFrame) -> pd.Series:
# # # # #         """
# # # # #         Funtion to run the decision tree on test inputs
# # # # #         """

# # # # #         # Traverse the tree you constructed to return the predicted values for the given test inputs.

# # # # #         pass

# # # # #     def plot(self) -> None:
# # # # #         """
# # # # #         Function to plot the tree

# # # # #         Output Example:
# # # # #         ?(X1 > 4)
# # # # #             Y: ?(X2 > 7)
# # # # #                 Y: Class A
# # # # #                 N: Class B
# # # # #             N: Class C
# # # # #         Where Y => Yes and N => No
# # # # #         """
# # # # #         pass



# # # # """
# # # # Decision Tree supporting:
# # # # - Discrete input (via one-hot to real), discrete output (classification)
# # # # - Real input, discrete output (classification)
# # # # - Real input, real output (regression)
# # # # - Discrete input (via one-hot), real output (regression)
# # # # Splitting:
# # # # - Classification: entropy (information_gain) or gini_index
# # # # - Regression: MSE reduction
# # # # """

# # # # from __future__ import annotations
# # # # from dataclasses import dataclass
# # # # from typing import Literal, Optional

# # # # import numpy as np
# # # # import pandas as pd

# # # # from tree.utils import (
# # # #     one_hot_encoding,
# # # #     check_ifreal,
# # # #     opt_split_attribute,
# # # #     split_data,
# # # # )


# # # # np.random.seed(42)


# # # # @dataclass
# # # # class _Node:
# # # #     is_leaf: bool
# # # #     prediction: Optional[float] = None  # for regression or class id
# # # #     feature: Optional[str] = None
# # # #     threshold: Optional[float] = None
# # # #     left: Optional["_Node"] = None
# # # #     right: Optional["_Node"] = None
# # # #     depth: int = 0
# # # #     n_samples: int = 0


# # # # class DecisionTree:
# # # #     criterion: Literal["information_gain", "gini_index"]  # (classification only)
# # # #     max_depth: int

# # # #     def __init__(self, criterion="information_gain", max_depth=5):
# # # #         if criterion not in ("information_gain", "gini_index"):
# # # #             raise ValueError("criterion must be 'information_gain' or 'gini_index'")
# # # #         self.criterion = criterion
# # # #         self.max_depth = int(max_depth)
# # # #         self.root_: Optional[_Node] = None
# # # #         self.task_: Optional[str] = None  # "classification" or "regression"
# # # #         self.features_: Optional[list[str]] = None

# # # #     # ----------- Public API ----------- #
# # # #     def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
# # # #         """
# # # #         Train/construct the decision tree.
# # # #         Automatically one-hot encodes categorical inputs.
# # # #         """
# # # #         assert isinstance(X, pd.DataFrame), "X must be a pandas DataFrame"
# # # #         assert isinstance(y, pd.Series), "y must be a pandas Series"
# # # #         assert len(X) == len(y), "X and y must have same length"
# # # #         assert self.max_depth >= 0, "max_depth must be >= 0"

# # # #         # Determine task
# # # #         self.task_ = "regression" if check_ifreal(y) else "classification"

# # # #         # One-hot encode inputs if necessary; ensure float dtype
# # # #         X_proc = one_hot_encoding(X)
# # # #         self.features_ = list(X_proc.columns)

# # # #         # Build the tree
# # # #         self.root_ = self._build_tree(X_proc, y, depth=0)

# # # #     def predict(self, X: pd.DataFrame) -> pd.Series:
# # # #         """
# # # #         Predict using the fitted tree.
# # # #         We apply the same one-hot encoding columns as seen in training (missing columns default to 0).
# # # #         """
# # # #         assert self.root_ is not None, "Call fit() before predict()"
# # # #         assert isinstance(X, pd.DataFrame)

# # # #         # Apply one-hot with training-detected columns
# # # #         X_proc = one_hot_encoding(X)
# # # #         # Align columns: add missing columns with 0, drop extras
# # # #         for c in self.features_:
# # # #             if c not in X_proc.columns:
# # # #                 X_proc[c] = 0.0
# # # #         X_proc = X_proc[self.features_]

# # # #         preds = []
# # # #         for i in range(len(X_proc)):
# # # #             xrow = X_proc.iloc[i]
# # # #             preds.append(self._predict_one(xrow, self.root_))

# # # #         # Return dtype: classification returns category (if train y was category/int), regression float
# # # #         if self.task_ == "classification":
# # # #             return pd.Series(preds).astype(type(preds[0]) if len(preds) else int)
# # # #         else:
# # # #             return pd.Series(preds, dtype=float)

# # # #     def plot(self) -> None:
# # # #         """
# # # #         Pretty-print the tree as text.
# # # #         """
# # # #         assert self.root_ is not None, "Tree not fitted."
# # # #         lines = []
# # # #         self._render(self.root_, prefix="", is_left=None, out=lines)
# # # #         print("\n".join(lines))

# # # #     # ----------- Internal helpers ----------- #
# # # #     def _leaf_value(self, y: pd.Series):
# # # #         if self.task_ == "classification":
# # # #             # Majority class; break ties deterministically
# # # #             vc = y.value_counts()
# # # #             top = vc.index[vc.values == vc.max()]
# # # #             # return top.sort_values().iloc[0]
# # # #             # return top.index[0]
# # # #             return y.value_counts().idxmax()
# # # #             # return y.mean()


# # # #         else:
# # # #             return float(y.mean())

# # # #     def _build_tree(self, X: pd.DataFrame, y: pd.Series, depth: int) -> _Node:
# # # #         node = _Node(is_leaf=False, depth=depth, n_samples=len(y))

# # # #         # Stopping conditions
# # # #         if depth >= self.max_depth:
# # # #             node.is_leaf = True
# # # #             node.prediction = self._leaf_value(y)
# # # #             return node

# # # #         # If pure (classification) or no variance-reducing split (handled later)
# # # #         if self.task_ == "classification" and y.nunique() == 1:
# # # #             node.is_leaf = True
# # # #             node.prediction = y.iloc[0]
# # # #             return node
# # # #         if len(X.columns) == 0 or len(y) <= 1:
# # # #             node.is_leaf = True
# # # #             node.prediction = self._leaf_value(y)
# # # #             return node

# # # #         # Choose best feature & threshold
# # # #         feat, thr, gain = opt_split_attribute(X, y, self.criterion, features=self.features_)
# # # #         if feat is None or gain <= 1e-12:
# # # #             node.is_leaf = True
# # # #             node.prediction = self._leaf_value(y)
# # # #             return node

# # # #         # Split
# # # #         Xl, yl, Xr, yr = split_data(X, y, feat, thr)
# # # #         if len(yl) == 0 or len(yr) == 0:
# # # #             node.is_leaf = True
# # # #             node.prediction = self._leaf_value(y)
# # # #             return node

# # # #         node.feature = feat
# # # #         node.threshold = thr
# # # #         node.left = self._build_tree(Xl, yl, depth + 1)
# # # #         node.right = self._build_tree(Xr, yr, depth + 1)
# # # #         return node

# # # #     def _predict_one(self, xrow: pd.Series, node: _Node):
# # # #         while not node.is_leaf:
# # # #             # Safe defaults if something odd happens
# # # #             f = node.feature
# # # #             t = node.threshold
# # # #             if f is None or t is None:
# # # #                 break
# # # #             if float(xrow[f]) <= t:
# # # #                 node = node.left
# # # #             else:
# # # #                 node = node.right
# # # #             if node is None:
# # # #                 # Fallback in case of misalignment
# # # #                 return 0 if self.task_ == "classification" else 0.0
# # # #         return node.prediction

# # # #     def _render(self, node: _Node, prefix: str, is_left: Optional[bool], out: list[str]):
# # # #         connector = ""
# # # #         if is_left is True:
# # # #             connector = "Y: "
# # # #         elif is_left is False:
# # # #             connector = "N: "
# # # #         if node.is_leaf:
# # # #             label = f"{connector}Leaf(n={node.n_samples}): {node.prediction}"
# # # #             out.append(prefix + label)
# # # #         else:
# # # #             question = f"?({node.feature} <= {node.threshold:.4f})"
# # # #             out.append(prefix + connector + question)
# # # #             self._render(node.left, prefix + "    ", True, out)
# # # #             self._render(node.right, prefix + "    ", False, out)




# # # tree/base.py
# # from dataclasses import dataclass
# # from typing import Literal, Optional, Any
# # import numpy as np
# # import pandas as pd
# # import matplotlib.pyplot as plt

# # from tree.utils import (
# #     one_hot_encoding,
# #     check_ifreal,
# #     opt_split_attribute,
# #     split_data,
# # )

# # np.random.seed(42)


# # @dataclass
# # class DecisionTree:
# #     criterion: Literal["information_gain", "gini_index"]  # classification criterion; ignored for regression
# #     max_depth: int  # maximum depth

# #     def __init__(self, criterion="information_gain", max_depth=5):
# #         self.criterion = criterion
# #         self.max_depth = max_depth
# #         self.root = None
# #         self.is_regression = None
# #         self.feature_names = None
# #         self._original_X_columns = None

# #     def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
# #         """
# #         Train and construct the decision tree.

# #         Strategy:
# #         - Convert categorical input columns to one-hot numeric via one_hot_encoding.
# #         - Determine if the problem is regression or classification using check_ifreal(y).
# #         - Recursively build a binary tree using opt_split_attribute (which evaluates thresholds).
# #         """
# #         if not isinstance(X, pd.DataFrame):
# #             raise ValueError("X must be a pandas DataFrame")
# #         if not isinstance(y, pd.Series):
# #             y = pd.Series(y)

# #         # store column names for later reference
# #         self._original_X_columns = X.columns.tolist()

# #         # Convert categorical inputs to numeric one-hot columns
# #         X_proc = one_hot_encoding(X)

# #         self.feature_names = X_proc.columns.tolist()
# #         self.is_regression = check_ifreal(y)

# #         # build tree recursively
# #         def build_tree(X_sub: pd.DataFrame, y_sub: pd.Series, depth: int):
# #             # Stopping conditions
# #             node = {}
# #             # if no samples
# #             if X_sub.shape[0] == 0:
# #                 node["is_leaf"] = True
# #                 node["prediction"] = None
# #                 return node

# #             # If regression -> leaf prediction = mean
# #             if self.is_regression:
# #                 # If all values almost equal or depth exceeded
# #                 if depth >= self.max_depth or y_sub.nunique() == 1 or X_sub.shape[0] <= 1:
# #                     node["is_leaf"] = True
# #                     node["prediction"] = float(y_sub.mean())
# #                     return node
# #             else:
# #                 # classification -> leaf prediction = mode
# #                 if depth >= self.max_depth or y_sub.nunique() == 1 or X_sub.shape[0] <= 1:
# #                     node["is_leaf"] = True
# #                     # mode may return a Series; pick first
# #                     node["prediction"] = y_sub.mode().iloc[0]
# #                     return node

# #             # find best split
# #             best_feature, best_threshold, best_gain = opt_split_attribute(
# #                 X_sub, y_sub, self.criterion, X_sub.columns
# #             )

# #             # If no valid split found -> leaf
# #             if best_feature is None or best_gain <= 1e-9:
# #                 node["is_leaf"] = True
# #                 if self.is_regression:
# #                     node["prediction"] = float(y_sub.mean())
# #                 else:
# #                     node["prediction"] = y_sub.mode().iloc[0]
# #                 return node

# #             # create decision node
# #             node["is_leaf"] = False
# #             node["feature"] = best_feature
# #             node["threshold"] = best_threshold

# #             # split and recurse
# #             X_left, y_left, X_right, y_right = split_data(X_sub, y_sub, best_feature, best_threshold)
# #             node["left"] = build_tree(X_left, y_left, depth + 1)
# #             node["right"] = build_tree(X_right, y_right, depth + 1)
# #             return node

# #         self.root = build_tree(X_proc, y, depth=0)
# #         # keep the processed X column names so predict can also process inputs
# #         self._processed_feature_names = self.feature_names

# #     def _predict_row(self, row: pd.Series, node: dict):
# #         # Traverses the tree for a single row (row must include processed one-hot columns).
# #         if node is None:
# #             return None
# #         if node.get("is_leaf", False):
# #             return node.get("prediction", None)
# #         feature = node["feature"]
# #         threshold = node["threshold"]
# #         # If input row missing the feature (possible if column didn't exist) treat as False -> go to right
# #         val = row.get(feature, 0.0)
# #         if pd.isna(val):
# #             # default to 0
# #             val = 0.0
# #         if val <= threshold:
# #             return self._predict_row(row, node["left"])
# #         else:
# #             return self._predict_row(row, node["right"])

# #     def predict(self, X: pd.DataFrame) -> pd.Series:
# #         """
# #         Predict for X. Must do the same one-hot encoding transformation as during training.
# #         Returns a pandas Series.
# #         """
# #         if self.root is None:
# #             raise ValueError("Tree has not been fitted")

# #         X_proc = one_hot_encoding(X)

# #         # Ensure all processed columns from training exist in X_proc; add missing columns with zeros
# #         for col in self._processed_feature_names:
# #             if col not in X_proc.columns:
# #                 X_proc[col] = 0.0
# #         # Extra columns in X_proc that were not present in training are ignored.

# #         # Keep same column order
# #         X_proc = X_proc[self._processed_feature_names]

# #         preds = []
# #         for _, row in X_proc.iterrows():
# #             p = self._predict_row(row, self.root)
# #             preds.append(p)

# #         # Return a Series. For regression, values are floats. For classification, labels may be of original dtype.
# #         return pd.Series(preds, index=X.index)

# #     def _print_tree(self, node: dict, depth: int = 0, branch: str = ""):
# #         indent = "    " * depth
# #         if node is None:
# #             print(f"{indent}{branch} None")
# #             return
# #         if node.get("is_leaf", False):
# #             print(f"{indent}{branch}Leaf: Predict -> {node.get('prediction')}")
# #             return
# #         feat = node["feature"]
# #         thr = node["threshold"]
# #         # print decision
# #         print(f"{indent}{branch}?({feat} <= {thr:.4f})")
# #         # Left branch labelled 'Y' (Yes: <=)
# #         self._print_tree(node["left"], depth + 1, branch="Y: ")
# #         # Right branch labelled 'N' (No: >)
# #         self._print_tree(node["right"], depth + 1, branch="N: ")

# #     def plot(self) -> None:
# #         """
# #         Print a textual representation of the tree to stdout.
# #         """
# #         if self.root is None:
# #             print("Empty tree (not fitted).")
# #             return
# #         self._print_tree(self.root)




# # # # tree/base.py
# # # from dataclasses import dataclass
# # # from typing import Literal, Optional, Any
# # # import numpy as np
# # # import pandas as pd
# # # import matplotlib.pyplot as plt

# # # from tree.utils import (
# # #     one_hot_encoding,
# # #     check_ifreal,
# # #     opt_split_attribute,
# # #     split_data,
# # # )

# # # np.random.seed(42)


# # # @dataclass
# # # class DecisionTree:
# # #     criterion: Literal["information_gain", "gini_index"]  # classification criterion; ignored for regression
# # #     max_depth: int  # maximum depth

# # #     def __init__(self, criterion="information_gain", max_depth=5):
# # #         self.criterion = criterion
# # #         self.max_depth = max_depth
# # #         self.root = None
# # #         self.is_regression = None
# # #         self.feature_names = None
# # #         self._original_X_columns = None
# # #         self._processed_feature_names = None

# # #     def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
# # #         """
# # #         Train and construct the decision tree.

# # #         Strategy:
# # #         - Convert categorical input columns to one-hot numeric via one_hot_encoding.
# # #         - Determine if the problem is regression or classification using check_ifreal(y).
# # #         - Recursively build a binary tree using opt_split_attribute (which evaluates thresholds).
# # #         """
# # #         if not isinstance(X, pd.DataFrame):
# # #             raise ValueError("X must be a pandas DataFrame")
# # #         if not isinstance(y, pd.Series):
# # #             y = pd.Series(y)

# # #         # store column names for later reference
# # #         self._original_X_columns = X.columns.tolist()

# # #         # Convert categorical inputs to numeric one-hot columns
# # #         X_proc = one_hot_encoding(X)

# # #         self.feature_names = X_proc.columns.tolist()
# # #         self.is_regression = check_ifreal(y)

# # #         # build tree recursively
# # #         def build_tree(X_sub: pd.DataFrame, y_sub: pd.Series, depth: int):
# # #             # Stopping conditions
# # #             node = {}
# # #             # if no samples
# # #             if X_sub.shape[0] == 0:
# # #                 node["is_leaf"] = True
# # #                 node["prediction"] = None
# # #                 return node

# # #             # If regression -> leaf prediction = mean
# # #             if self.is_regression:
# # #                 # If all values almost equal or depth exceeded
# # #                 if depth >= self.max_depth or y_sub.nunique() == 1 or X_sub.shape[0] <= 1:
# # #                     node["is_leaf"] = True
# # #                     node["prediction"] = float(y_sub.mean())
# # #                     return node
# # #             else:
# # #                 # classification -> leaf prediction = mode
# # #                 if depth >= self.max_depth or y_sub.nunique() == 1 or X_sub.shape[0] <= 1:
# # #                     node["is_leaf"] = True
# # #                     # mode may return a Series; pick first
# # #                     node["prediction"] = y_sub.mode().iloc[0]
# # #                     return node

# # #             # find best split
# # #             best_feature, best_threshold, best_gain = opt_split_attribute(
# # #                 X_sub, y_sub, self.criterion, X_sub.columns
# # #             )

# # #             # If no valid split found -> leaf
# # #             if best_feature is None or best_gain <= 1e-9:
# # #                 node["is_leaf"] = True
# # #                 if self.is_regression:
# # #                     node["prediction"] = float(y_sub.mean())
# # #                 else:
# # #                     node["prediction"] = y_sub.mode().iloc[0]
# # #                 return node

# # #             # create decision node
# # #             node["is_leaf"] = False
# # #             node["feature"] = best_feature
# # #             node["threshold"] = best_threshold

# # #             # split and recurse
# # #             X_left, y_left, X_right, y_right = split_data(X_sub, y_sub, best_feature, best_threshold)
# # #             node["left"] = build_tree(X_left, y_left, depth + 1)
# # #             node["right"] = build_tree(X_right, y_right, depth + 1)
# # #             return node

# # #         self.root = build_tree(X_proc, y, depth=0)
# # #         # keep the processed X column names so predict can also process inputs
# # #         self._processed_feature_names = self.feature_names

# # #     def _predict_row(self, row: pd.Series, node: dict):
# # #         # Traverses the tree for a single row (row must include processed one-hot columns).
# # #         if node is None:
# # #             return None
# # #         if node.get("is_leaf", False):
# # #             return node.get("prediction", None)
# # #         feature = node["feature"]
# # #         threshold = node["threshold"]
# # #         # If input row missing the feature (possible if column didn't exist) treat as 0.0
# # #         val = row.get(feature, 0.0)
# # #         if pd.isna(val):
# # #             val = 0.0
# # #         if val <= threshold:
# # #             return self._predict_row(row, node["left"])
# # #         else:
# # #             return self._predict_row(row, node["right"])

# # #     def predict(self, X: pd.DataFrame) -> pd.Series:
# # #         """
# # #         Predict for X. Must do the same one-hot encoding transformation as during training.
# # #         Returns a pandas Series.
# # #         """
# # #         if self.root is None:
# # #             raise ValueError("Tree has not been fitted")

# # #         X_proc = one_hot_encoding(X)

# # #         # Ensure all processed columns from training exist in X_proc; add missing columns with zeros
# # #         for col in self._processed_feature_names:
# # #             if col not in X_proc.columns:
# # #                 X_proc[col] = 0.0
# # #         # Extra columns in X_proc that were not present in training are ignored.

# # #         # Keep same column order
# # #         X_proc = X_proc[self._processed_feature_names]

# # #         preds = []
# # #         for _, row in X_proc.iterrows():
# # #             p = self._predict_row(row, self.root)
# # #             preds.append(p)

# # #         # Return a Series. For regression, values are floats. For classification, labels may be of original dtype.
# # #         return pd.Series(preds, index=X.index)

# # #     def _print_tree(self, node: dict, depth: int = 0, branch: str = ""):
# # #         indent = "    " * depth
# # #         if node is None:
# # #             print(f"{indent}{branch} None")
# # #             return
# # #         if node.get("is_leaf", False):
# # #             print(f"{indent}{branch}Leaf: Predict -> {node.get('prediction')}")
# # #             return
# # #         feat = node["feature"]
# # #         thr = node["threshold"]
# # #         # print decision
# # #         print(f"{indent}{branch}?({feat} <= {thr:.4f})")
# # #         # Left branch labelled 'Y' (Yes: <=)
# # #         self._print_tree(node["left"], depth + 1, branch="Y: ")
# # #         # Right branch labelled 'N' (No: >)
# # #         self._print_tree(node["right"], depth + 1, branch="N: ")

# # #     # ------------------ new plotting helpers ------------------ #
# # #     def _collect_nodes(self, node: dict, nodes: list, parent: Optional[dict] = None):
# # #         """
# # #         Collect nodes in pre-order and link parents.
# # #         Each entry is a tuple (node_dict, parent_node_dict or None).
# # #         """
# # #         if node is None:
# # #             return
# # #         nodes.append((node, parent))
# # #         if not node.get("is_leaf", False):
# # #             self._collect_nodes(node["left"], nodes, node)
# # #             self._collect_nodes(node["right"], nodes, node)

# # #     def _assign_positions(self, node: dict, depth: int = 0, x_counter: list = None, pos: dict = None, depth_map: dict = None):
# # #         """
# # #         Assign x,y positions for each node (id(node) used as key).
# # #         We assign x positions by doing an in-order leaf-based layout:
# # #           - if leaf: x = next counter; y = -depth
# # #           - else: x = mean(x_left, x_right); y = -depth
# # #         x_counter is a single-element list used as mutable integer across recursion.
# # #         """
# # #         if x_counter is None:
# # #             x_counter = [0]
# # #         if pos is None:
# # #             pos = {}
# # #         if depth_map is None:
# # #             depth_map = {}
# # #         if node.get("is_leaf", False):
# # #             x = x_counter[0]
# # #             pos[id(node)] = (x, -depth)
# # #             depth_map[id(node)] = depth
# # #             x_counter[0] += 1
# # #             return pos, depth_map
# # #         # internal node
# # #         # compute left subtree
# # #         pos, depth_map = self._assign_positions(node["left"], depth + 1, x_counter, pos, depth_map)
# # #         # compute right subtree
# # #         pos, depth_map = self._assign_positions(node["right"], depth + 1, x_counter, pos, depth_map)
# # #         # now set this node's x to mean of children
# # #         left_pos = pos[id(node["left"])]
# # #         right_pos = pos[id(node["right"])]
# # #         x = (left_pos[0] + right_pos[0]) / 2.0
# # #         pos[id(node)] = (x, -depth)
# # #         depth_map[id(node)] = depth
# # #         return pos, depth_map

# # #     def _node_label(self, node: dict) -> str:
# # #         if node.get("is_leaf", False):
# # #             pred = node.get("prediction", None)
# # #             # format floats nicely
# # #             if isinstance(pred, float):
# # #                 return f"Leaf:\\n{pred:.6f}"
# # #             else:
# # #                 return f"Leaf:\\n{pred}"
# # #         else:
# # #             feat = node["feature"]
# # #             thr = node["threshold"]
# # #             return f"?({feat} <= {thr:.4f})"

# # #     def plot(self, filename: str = "tree.png", dpi: int = 200) -> None:
# # #         """
# # #         Create a PNG image of the decision tree and save as `filename`.
# # #         Also prints the textual tree to stdout (same as before).
# # #         """
# # #         # Print textual tree as before
# # #         if self.root is None:
# # #             print("Empty tree (not fitted).")
# # #             return
# # #         self._print_tree(self.root)

# # #         # Collect nodes and assign positions
# # #         pos, depth_map = self._assign_positions(self.root)

# # #         # Create figure sized based on breadth and depth
# # #         # width ~ number of leaves, height ~ depth
# # #         xs = [p[0] for p in pos.values()]
# # #         ys = [p[1] for p in pos.values()]
# # #         if len(xs) == 0:
# # #             print("No nodes to plot.")
# # #             return
# # #         min_x, max_x = min(xs), max(xs)
# # #         min_y, max_y = min(ys), max(ys)
# # #         # compute figure size heuristically
# # #         width = max(6.0, (max_x - min_x + 1) * 0.8)
# # #         height = max(4.0, (abs(min_y - max_y) + 1) * 0.9)

# # #         fig, ax = plt.subplots(figsize=(width, height))
# # #         ax.set_axis_off()

# # #         # transform positions to nicer coordinates: scale x to [0, 1], y to [0, 1]
# # #         # but keep relative spacing
# # #         x_vals = xs
# # #         y_vals = ys
# # #         x_span = max_x - min_x if max_x > min_x else 1.0
# # #         y_span = abs(min_y - max_y) if min_y < max_y else abs(min_y) + 1.0

# # #         # padding
# # #         pad_x = 0.5
# # #         pad_y = 0.5

# # #         # map function
# # #         def map_x(x):
# # #             return (x - min_x) / x_span * (width - pad_x * 2) + pad_x

# # #         def map_y(y):
# # #             # y is negative depths; map deeper (more negative) lower on canvas
# # #             return (y - max_y) / (min_y - max_y if min_y != max_y else 1.0) * (height - pad_y * 2) + pad_y

# # #         # Draw edges first (so nodes are on top)
# # #         def draw_edges(node: dict):
# # #             if node.get("is_leaf", False):
# # #                 return
# # #             parent_xy = pos[id(node)]
# # #             px = map_x(parent_xy[0])
# # #             py = map_y(parent_xy[1])
# # #             # left
# # #             left = node["left"]
# # #             right = node["right"]
# # #             lx, ly = pos[id(left)]
# # #             rx, ry = pos[id(right)]
# # #             lx = map_x(lx); ly = map_y(ly)
# # #             rx = map_x(rx); ry = map_y(ry)
# # #             # draw lines
# # #             ax.plot([px, lx], [py, ly], linewidth=1, zorder=1)
# # #             ax.plot([px, rx], [py, ry], linewidth=1, zorder=1)
# # #             # recurse
# # #             draw_edges(left)
# # #             draw_edges(right)

# # #         draw_edges(self.root)

# # #         # Draw nodes (boxes with text)
# # #         def draw_nodes(node: dict):
# # #             node_xy = pos[id(node)]
# # #             x = map_x(node_xy[0])
# # #             y = map_y(node_xy[1])
# # #             label = self._node_label(node)
# # #             bbox = dict(boxstyle="round,pad=0.4", facecolor="#ffffff", edgecolor="black", linewidth=0.8)
# # #             ax.text(x, y, label, ha="center", va="center", fontsize=8, bbox=bbox, zorder=2)
# # #             if not node.get("is_leaf", False):
# # #                 draw_nodes(node["left"])
# # #                 draw_nodes(node["right"])

# # #         draw_nodes(self.root)

# # #         # save and show
# # #         plt.tight_layout()
# # #         fig.savefig(filename, dpi=dpi)
# # #         plt.close(fig)
# # #         print(f"Tree plotted and saved to {filename}")






# # tree/base.py
# from dataclasses import dataclass
# from typing import Literal, Optional, Any
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import shutil
# import subprocess
# import tempfile
# import os

# from tree.utils import (
#     one_hot_encoding,
#     check_ifreal,
#     opt_split_attribute,
#     split_data,
# )

# np.random.seed(42)


# @dataclass
# class DecisionTree:
#     criterion: Literal["information_gain", "gini_index"]  # classification criterion; ignored for regression
#     max_depth: int  # maximum depth

#     def __init__(self, criterion="information_gain", max_depth=5):
#         self.criterion = criterion
#         self.max_depth = max_depth
#         self.root = None
#         self.is_regression = None
#         self.feature_names = None
#         self._original_X_columns = None
#         self._processed_feature_names = None

#     def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
#         if not isinstance(X, pd.DataFrame):
#             raise ValueError("X must be a pandas DataFrame")
#         if not isinstance(y, pd.Series):
#             y = pd.Series(y)

#         self._original_X_columns = X.columns.tolist()
#         X_proc = one_hot_encoding(X)
#         self.feature_names = X_proc.columns.tolist()
#         self.is_regression = check_ifreal(y)

#         def build_tree(X_sub: pd.DataFrame, y_sub: pd.Series, depth: int):
#             node = {}
#             if X_sub.shape[0] == 0:
#                 node["is_leaf"] = True
#                 node["prediction"] = None
#                 return node

#             if self.is_regression:
#                 if depth >= self.max_depth or y_sub.nunique() == 1 or X_sub.shape[0] <= 1:
#                     node["is_leaf"] = True
#                     node["prediction"] = float(y_sub.mean())
#                     return node
#             else:
#                 if depth >= self.max_depth or y_sub.nunique() == 1 or X_sub.shape[0] <= 1:
#                     node["is_leaf"] = True
#                     node["prediction"] = y_sub.mode().iloc[0]
#                     return node

#             best_feature, best_threshold, best_gain = opt_split_attribute(
#                 X_sub, y_sub, self.criterion, X_sub.columns
#             )

#             if best_feature is None or best_gain <= 1e-9:
#                 node["is_leaf"] = True
#                 if self.is_regression:
#                     node["prediction"] = float(y_sub.mean())
#                 else:
#                     node["prediction"] = y_sub.mode().iloc[0]
#                 return node

#             node["is_leaf"] = False
#             node["feature"] = best_feature
#             node["threshold"] = best_threshold
#             X_left, y_left, X_right, y_right = split_data(X_sub, y_sub, best_feature, best_threshold)
#             node["left"] = build_tree(X_left, y_left, depth + 1)
#             node["right"] = build_tree(X_right, y_right, depth + 1)
#             return node

#         self.root = build_tree(X_proc, y, depth=0)
#         self._processed_feature_names = self.feature_names

#     def _predict_row(self, row: pd.Series, node: dict):
#         if node is None:
#             return None
#         if node.get("is_leaf", False):
#             return node.get("prediction", None)
#         feature = node["feature"]
#         threshold = node["threshold"]
#         val = row.get(feature, 0.0)
#         if pd.isna(val):
#             val = 0.0
#         if val <= threshold:
#             return self._predict_row(row, node["left"])
#         else:
#             return self._predict_row(row, node["right"])

#     def predict(self, X: pd.DataFrame) -> pd.Series:
#         if self.root is None:
#             raise ValueError("Tree has not been fitted")

#         X_proc = one_hot_encoding(X)
#         for col in self._processed_feature_names:
#             if col not in X_proc.columns:
#                 X_proc[col] = 0.0
#         X_proc = X_proc[self._processed_feature_names]

#         preds = []
#         for _, row in X_proc.iterrows():
#             p = self._predict_row(row, self.root)
#             preds.append(p)
#         return pd.Series(preds, index=X.index)

#     def _print_tree(self, node: dict, depth: int = 0, branch: str = ""):
#         indent = "    " * depth
#         if node is None:
#             print(f"{indent}{branch} None")
#             return
#         if node.get("is_leaf", False):
#             print(f"{indent}{branch}Leaf: Predict -> {node.get('prediction')}")
#             return
#         feat = node["feature"]
#         thr = node["threshold"]
#         print(f"{indent}{branch}?({feat} <= {thr:.4f})")
#         self._print_tree(node["left"], depth + 1, branch="Y: ")
#         self._print_tree(node["right"], depth + 1, branch="N: ")

#     # ---------------- DOT/Graphviz plotting ----------------- #
#     def _build_dot(self, node: dict, counter: list, lines: list):
#         """
#         Recursively build DOT lines. Use counter (single-element list) to assign numeric IDs.
#         Return node_id.
#         """
#         node_id = counter[0]
#         counter[0] += 1

#         if node.get("is_leaf", False):
#             pred = node.get("prediction", None)
#             if isinstance(pred, float):
#                 label = f"Leaf:\\n{pred:.6f}"
#             else:
#                 label = f"Leaf:\\n{pred}"
#             # use box with rounded corners
#             lines.append(f'  node{node_id} [label="{label}", shape=box, style="rounded,filled", fillcolor="#FFFFFF", fontsize=10];')
#             return node_id

#         # internal node
#         feat = node["feature"]
#         thr = node["threshold"]
#         label = f"?({feat} <= {thr:.4f})"
#         lines.append(f'  node{node_id} [label="{label}", shape=box, style="rounded,filled", fillcolor="#FFFFFF", fontsize=10];')

#         # left child
#         left_id = self._build_dot(node["left"], counter, lines)
#         # right child
#         right_id = self._build_dot(node["right"], counter, lines)

#         # edges: label left with Y, right with N
#         lines.append(f'  node{node_id} -> node{left_id} [label="Y", fontsize=9];')
#         lines.append(f'  node{node_id} -> node{right_id} [label="N", fontsize=9];')

#         return node_id

#     def plot(self, filename: Optional[str] = None, format: str = "png", engine: str = "dot") -> None:
#         """
#         Plot the tree using Graphviz (DOT). If filename is None a default name is used.
#         - Tries to use python graphviz (if installed)
#         - Falls back to calling system 'dot' if available
#         - If neither available, falls back to textual print + matplotlib preview
#         """
#         if self.root is None:
#             print("Empty tree (not fitted).")
#             return

#         # print textual tree (same as before)
#         self._print_tree(self.root)

#         # create DOT text
#         lines = ['digraph DecisionTree {', '  node [shape=box];', '  rankdir=TB;']
#         counter = [0]
#         self._build_dot(self.root, counter, lines)
#         lines.append('}')
#         dot_text = "\n".join(lines)

#         # set default filename if not provided
#         if filename is None:
#             # create name using criterion and a small random seed to avoid overwrite
#             filename = f"tree_{self.criterion}_{np.random.randint(1_000_000)}.{format}"

#         # First try using python graphviz if available
#         try:
#             import graphviz as gv  # type: ignore
#             src = gv.Source(dot_text)
#             outpath = src.render(filename=filename, format=format, cleanup=True)
#             print(f"Graphviz (python package) rendered tree to {outpath}")
#             return
#         except Exception as e:
#             # fall-through to try system dot
#             # print debug only if dot not found or other failure
#             # print("graphviz python package not available or failed:", e)
#             pass

#         # Try system 'dot' binary
#         dot_path = shutil.which("dot")
#         if dot_path:
#             try:
#                 # Use a temporary file to write the dot and call dot to render
#                 with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".dot") as f:
#                     f.write(dot_text)
#                     tmpdot = f.name
#                 out_png = filename if filename.endswith(f".{format}") else f"{filename}.{format}"
#                 subprocess.run([dot_path, f"-K{engine}", f"-T{format}", tmpdot, "-o", out_png], check=True)
#                 # cleanup temporary dot
#                 try:
#                     os.remove(tmpdot)
#                 except Exception:
#                     pass
#                 print(f"Graphviz (dot) rendered tree to {out_png}")
#                 return
#             except Exception as e:
#                 print("Failed to run system 'dot' for rendering:", e)

#         # Fallback: no graphviz available; simple matplotlib textual drawing
#         print("Graphviz not available — falling back to textual matplotlib plot (no PNG).")
#         # reuse the previous matplotlib textual plotting approach
#         self._matplotlib_fallback_plot(filename)

#     def _matplotlib_fallback_plot(self, filename: str):
#         # Simple textual 'plot' using matplotlib: just render printed tree to an image
#         fig, ax = plt.subplots(figsize=(12, 6))
#         ax.axis("off")
#         # get textual lines
#         import io, textwrap
#         buf = io.StringIO()
#         self._capture_print(self.root, buf)
#         text = buf.getvalue()
#         # draw text
#         wrapped = textwrap.fill(text, width=120)
#         ax.text(0.01, 0.99, text, va="top", ha="left", family="monospace", fontsize=8)
#         out = filename if filename.endswith(".png") else f"{filename}.png"
#         plt.tight_layout()
#         fig.savefig(out, dpi=200)
#         plt.close(fig)
#         print(f"Saved fallback textual tree to {out}")

#     def _capture_print(self, node: dict, buf, depth: int = 0, branch: str = ""):
#         indent = "    " * depth
#         if node is None:
#             buf.write(f"{indent}{branch} None\n")
#             return
#         if node.get("is_leaf", False):
#             buf.write(f"{indent}{branch}Leaf: Predict -> {node.get('prediction')}\n")
#             return
#         feat = node["feature"]
#         thr = node["threshold"]
#         buf.write(f"{indent}{branch}?({feat} <= {thr:.4f})\n")
#         self._capture_print(node["left"], buf, depth + 1, branch="Y: ")
#         self._capture_print(node["right"], buf, depth + 1, branch="N: ")




# tree/base.py
"""Decision Tree implementation for the assignment."""
from dataclasses import dataclass
from typing import Literal, Optional, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tree.utils import (
    one_hot_encoding,
    check_ifreal,
    opt_split_attribute,
    split_data,
)

np.random.seed(42)


@dataclass
class DecisionTree:
    criterion: Literal["information_gain", "gini_index"]  # classification criterion; ignored for regression
    max_depth: int = 5  # maximum depth

    def __init__(self, criterion="information_gain", max_depth=5):
        self.criterion = criterion
        self.max_depth = max_depth
        self.root = None
        self.is_regression = None
        self._original_X_columns = None
        self._processed_feature_names = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Fit tree on X (DataFrame) and y (Series).
        Converts categorical inputs to one-hot. Stores processed feature names for predict.
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame")
        if not isinstance(y, pd.Series):
            y = pd.Series(y)

        self._original_X_columns = list(X.columns)
        X_proc = one_hot_encoding(X)
        self._processed_feature_names = list(X_proc.columns)
        self.is_regression = check_ifreal(y)

        def build(X_sub: pd.DataFrame, y_sub: pd.Series, depth: int):
            node = {}
            if X_sub.shape[0] == 0:
                node["is_leaf"] = True
                node["prediction"] = None
                return node

            # stopping criteria
            if self.is_regression:
                if depth >= self.max_depth or y_sub.nunique() == 1 or X_sub.shape[0] <= 1:
                    node["is_leaf"] = True
                    node["prediction"] = float(y_sub.mean())
                    return node
            else:
                if depth >= self.max_depth or y_sub.nunique() == 1 or X_sub.shape[0] <= 1:
                    node["is_leaf"] = True
                    node["prediction"] = y_sub.mode().iloc[0]
                    return node

            # choose best split
            best_feat, best_thr, best_gain = opt_split_attribute(X_sub, y_sub, self.criterion, X_sub.columns)
            if best_feat is None or best_gain <= 1e-9:
                node["is_leaf"] = True
                node["prediction"] = float(y_sub.mean()) if self.is_regression else y_sub.mode().iloc[0]
                return node

            # create internal node
            node["is_leaf"] = False
            node["feature"] = best_feat
            node["threshold"] = best_thr

            X_left, y_left, X_right, y_right = split_data(X_sub, y_sub, best_feat, best_thr)
            node["left"] = build(X_left, y_left, depth + 1)
            node["right"] = build(X_right, y_right, depth + 1)
            return node

        self.root = build(X_proc, y, depth=0)

    def _predict_row(self, row: pd.Series, node: dict):
        if node is None:
            return None
        if node.get("is_leaf", False):
            return node.get("prediction", None)
        feat = node["feature"]
        thr = node["threshold"]
        val = row.get(feat, 0.0)
        if pd.isna(val):
            val = 0.0
        if val <= thr:
            return self._predict_row(row, node["left"])
        else:
            return self._predict_row(row, node["right"])

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Predict values for X. X is converted with same one-hot encoding schema as training.
        """
        if self.root is None:
            raise ValueError("Tree has not been fitted yet.")

        X_proc = one_hot_encoding(X)
        # ensure processed columns present
        for c in self._processed_feature_names:
            if c not in X_proc.columns:
                X_proc[c] = 0.0
        # keep same order
        X_proc = X_proc[self._processed_feature_names]

        preds = []
        for _, row in X_proc.iterrows():
            preds.append(self._predict_row(row, self.root))
        return pd.Series(preds, index=X.index)

    def _print_tree(self, node: dict, depth: int = 0, branch: str = ""):
        indent = "    " * depth
        if node is None:
            print(f"{indent}{branch} None")
            return
        if node.get("is_leaf", False):
            print(f"{indent}{branch}Leaf: Predict -> {node.get('prediction')}")
            return
        feat = node["feature"]
        thr = node["threshold"]
        print(f"{indent}{branch}?({feat} <= {thr:.4f})")
        self._print_tree(node["left"], depth + 1, branch="Y: ")
        self._print_tree(node["right"], depth + 1, branch="N: ")

    def plot(self, filename: Optional[str] = None, format: str = "svg"):
        """
        Print the tree textually to stdout (same format as sample output).
        If a filename is provided and a helper module `generate_graph_from_tree` exists,
        the function will try to save a Graphviz SVG/PDF for a nice visual.
        """
        if self.root is None:
            print("Empty tree (not fitted).")
            return
        # textual print
        self._print_tree(self.root)

        # optional Graphviz export if helper is available
        if filename is not None:
            try:
                # lazy import to avoid hard dependency
                from generate_graph_from_tree import generate_graph_from_tree
                # call helper that produces a better graph (SVG/PDF)
                generate_graph_from_tree(self, pd.DataFrame(columns=self._original_X_columns), pd.Series(dtype=object),
                                         out_filename=filename, format=format, criterion=self.criterion)
                print(f"Saved graph to {filename}")
            except Exception:
                # don't fail if graph export unavailable; textual print is the main requirement
                pass

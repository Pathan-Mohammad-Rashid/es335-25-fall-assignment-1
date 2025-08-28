"""
The current code given is for the Assignment 1.
You will be expected to use this to make trees for:
> discrete input, discrete output
> real input, real output
> real input, discrete output
> discrete input, real output
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *
from tree_visualizer import generate_graph_from_tree


np.random.seed(42)
# Test case 1
# Real Input and Real Output

N = 30
P = 5
X = pd.DataFrame(np.random.randn(N, P))
y = pd.Series(np.random.randn(N))


for criteria in ["information_gain", "gini_index"]:
    tree = DecisionTree(criterion=criteria)  # Split based on Inf. Gain
    tree.fit(X, y)
    y_hat = tree.predict(X)
    tree.plot()
    print("Criteria :", criteria)
    print("RMSE: ", rmse(y_hat, y))
    print("MAE: ", mae(y_hat, y))

from generate_graph_from_tree import generate_graph_from_tree
generate_graph_from_tree(tree, X, y, out_filename="tree_case1.png")


# Test case 2
# Real Input and Discrete Output

N = 30
P = 5
X = pd.DataFrame(np.random.randn(N, P))
y = pd.Series(np.random.randint(P, size=N), dtype="category")

for criteria in ["information_gain", "gini_index"]:
    tree = DecisionTree(criterion=criteria)  # Split based on Inf. Gain
    tree.fit(X, y)
    y_hat = tree.predict(X)
    tree.plot()
    print("Criteria :", criteria)
    print("Accuracy: ", accuracy(y_hat, y))
    for cls in y.unique():
        print("Precision: ", precision(y_hat, y, cls))
        print("Recall: ", recall(y_hat, y, cls))


from generate_graph_from_tree import generate_graph_from_tree
generate_graph_from_tree(tree, X, y, out_filename="tree_case2.png")


# Test case 3
# Discrete Input and Discrete Output

N = 30
P = 5
X = pd.DataFrame({i: pd.Series(np.random.randint(P, size=N), dtype="category") for i in range(5)})
y = pd.Series(np.random.randint(P, size=N), dtype="category")

for criteria in ["information_gain", "gini_index"]:
    tree = DecisionTree(criterion=criteria)  # Split based on Inf. Gain
    tree.fit(X, y)
    y_hat = tree.predict(X)
    tree.plot()
    print("Criteria :", criteria)
    print("Accuracy: ", accuracy(y_hat, y))
    for cls in y.unique():
        print("Precision: ", precision(y_hat, y, cls))
        print("Recall: ", recall(y_hat, y, cls))

from generate_graph_from_tree import generate_graph_from_tree
generate_graph_from_tree(tree, X, y, out_filename="tree_case3.png")


# Test case 4
# Discrete Input and Real Output

N = 30
P = 5
X = pd.DataFrame({i: pd.Series(np.random.randint(P, size=N), dtype="category") for i in range(5)})
y = pd.Series(np.random.randn(N))

for criteria in ["information_gain", "gini_index"]:
    tree = DecisionTree(criterion=criteria)  # Split based on Inf. Gain
    tree.fit(X, y)
    y_hat = tree.predict(X)
    tree.plot()
    print("Criteria :", criteria)
    print("RMSE: ", rmse(y_hat, y))
    print("MAE: ", mae(y_hat, y))

from generate_graph_from_tree import generate_graph_from_tree
generate_graph_from_tree(tree, X, y, out_filename="tree_case4.png")


# from tree.utils import export_tree_png

# # # save tree plot
# # export_tree_png(tree.root_, "decision_tree")


# export_tree_png(
#     tree.root_,
#     "decision_tree_regression_infogain",
#     metrics={"Criteria": "information_gain", "RMSE": rmse, "MAE": mae}
# )

# export_tree_png(
#     tree.root_,
#     "decision_tree_classification_gini",
#     metrics={"Criteria": "gini_index", "Accuracy": accuracy}
# )

# metrics = {"Criteria": "information_gain", "Accuracy": accuracy}
# # for i, (p, r) in enumerate(zip(precision, recall)):
# #     metrics[f"Class {i} Precision"] = round(p, 3)
# #     metrics[f"Class {i} Recall"] = round(r, 3)

# export_tree_png(tree.root_, "decision_tree_classification_infogain", metrics=metrics)

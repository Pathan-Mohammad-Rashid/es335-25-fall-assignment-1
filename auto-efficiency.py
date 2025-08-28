# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from tree.base import DecisionTree
# from metrics import *

# np.random.seed(42)

# # Reading the data
# url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
# data = pd.read_csv(url, delim_whitespace=True, header=None,
#                  names=["mpg", "cylinders", "displacement", "horsepower", "weight",
#                         "acceleration", "model year", "origin", "car name"])

# # Clean the above data by removing redundant columns and rows with junk values
# # Compare the performance of your model with the decision tree module from scikit learn


import numpy as np
import pandas as pd
from tree.base import DecisionTree
from metrics import *
import matplotlib.pyplot as plt

# For comparison only (allowed by assignment for this part)
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

np.random.seed(42)

# Read data
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
data = pd.read_csv(
    url,
    delim_whitespace=True,
    header=None,
    names=["mpg", "cylinders", "displacement", "horsepower", "weight",
           "acceleration", "model year", "origin", "car name"],
    na_values=["?"]
)

# Clean:
# - Drop rows with missing horsepower (NaN after coercion)
# - Drop irrelevant "car name"
# - mpg is the target (regression)
data = data.dropna(subset=["horsepower"]).copy()
data["horsepower"] = pd.to_numeric(data["horsepower"])
data = data.drop(columns=["car name"])

# Features/target
y = data["mpg"]
X = data.drop(columns=["mpg"])

# Some features are discrete-ish integers but we allow the tree to one-hot them if beneficial.
# Split train/test
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)

# Our tree (regression: criterion value is ignored at split metric, but required by ctor)
my_tree = DecisionTree(criterion="information_gain", max_depth=6)
my_tree.fit(X_tr, y_tr)
y_hat = my_tree.predict(X_te)

print("=== Automotive Efficiency: Our DecisionTree (Regression) ===")
print("Test RMSE:", rmse(y_hat, y_te))
print("Test MAE :", mae(y_hat, y_te))
print()
# Optional: Show structure
# my_tree.plot()

# Compare with sklearn's DecisionTreeRegressor
sk_tree = DecisionTreeRegressor(random_state=42, max_depth=6, criterion="squared_error")
sk_tree.fit(pd.get_dummies(X_tr), y_tr)  # ensure fair encoding
sk_pred = sk_tree.predict(pd.get_dummies(X_te).reindex(columns=pd.get_dummies(X_tr).columns, fill_value=0))

print("=== scikit-learn DecisionTreeRegressor ===")
print("Test RMSE:", rmse(pd.Series(sk_pred), y_te))
print("Test MAE :", mae(pd.Series(sk_pred), y_te))

# You should copy results and brief analysis into: Asst#auto-efficiency_Q1.md and Asst#auto-efficiency_Q2.md

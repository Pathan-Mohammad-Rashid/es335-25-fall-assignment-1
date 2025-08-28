# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from tree.base import DecisionTree
# from metrics import *
# from sklearn.datasets import make_classification

# np.random.seed(42)

# # Code given in the question
# X, y = make_classification(
#     n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=2, class_sep=0.5)

# # For plotting
# plt.scatter(X[:, 0], X[:, 1], c=y)

# # Write the code for Q2 a) and b) below. Show your results.



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *
from sklearn.datasets import make_classification

np.random.seed(42)

# Generate data per spec
X_np, y_np = make_classification(
    n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=2, class_sep=0.5
)

# Plot
plt.figure()
plt.scatter(X_np[:, 0], X_np[:, 1], c=y_np)
plt.title("Synthetic classification data")
plt.xlabel("x0")
plt.ylabel("x1")
plt.show()

X = pd.DataFrame(X_np, columns=["x0", "x1"])
# categorical is optional; ints are fine for classes. We'll keep them as ints.
y = pd.Series(y_np)

# ---------- (a) 70/30 train-test split and per-class metrics ----------
N = len(X)
idx = np.arange(N)
np.random.shuffle(idx)

split = int(0.7 * N)
train_idx, test_idx = idx[:split], idx[split:]

X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
X_te, y_te = X.iloc[test_idx], y.iloc[test_idx]

tree = DecisionTree(criterion="information_gain", max_depth=5)
tree.fit(X_tr, y_tr)
y_hat = tree.predict(X_te)

print("=== Q2(a): 70/30 holdout results ===")
print(f"Accuracy: {accuracy(y_hat, y_te):.4f}")
for cls in sorted(y.unique()):
    p = precision(y_hat, y_te, cls)
    r = recall(y_hat, y_te, cls)
    print(f"Class {cls}: Precision={p:.4f}, Recall={r:.4f}")
print()
# (optional) visualize learnt tree
tree.plot()

# ---------- (b) 5-fold nested cross-validation to pick optimal depth ----------
def kfold_indices(n, k, rng):
    idx = np.arange(n)
    rng.shuffle(idx)
    folds = np.array_split(idx, k)
    return folds

def evaluate_depth(Xdf, yser, depth, criterion, folds_outer=5, folds_inner=4, rng=None):
    if rng is None:
        rng = np.random.RandomState(123)

    # Outer CV for unbiased estimate
    outer_folds = kfold_indices(len(Xdf), folds_outer, rng)
    outer_accs = []
    best_depths = []

    for outer_i in range(folds_outer):
        test_idx = outer_folds[outer_i]
        train_idx = np.concatenate([outer_folds[j] for j in range(folds_outer) if j != outer_i])
        X_train, y_train = Xdf.iloc[train_idx], yser.iloc[train_idx]
        X_test, y_test = Xdf.iloc[test_idx], yser.iloc[test_idx]

        # Inner CV to select depth
        inner_folds = kfold_indices(len(X_train), folds_inner, rng)
        candidate_depths = list(range(1, 9))  # search 1..8
        inner_scores = {d: [] for d in candidate_depths}
        for d in candidate_depths:
            for inner_i in range(folds_inner):
                val_idx = inner_folds[inner_i]
                tr_idx = np.concatenate([inner_folds[j] for j in range(folds_inner) if j != inner_i])
                X_tr2, y_tr2 = X_train.iloc[tr_idx], y_train.iloc[tr_idx]
                X_va2, y_va2 = X_train.iloc[val_idx], y_train.iloc[val_idx]
                model = DecisionTree(criterion=criterion, max_depth=d)
                model.fit(X_tr2, y_tr2)
                y_pred_va = model.predict(X_va2)
                inner_scores[d].append(accuracy(y_pred_va, y_va2))

        # Pick best depth
        avg_scores = {d: np.mean(inner_scores[d]) for d in candidate_depths}
        best_d = max(avg_scores, key=avg_scores.get)
        best_depths.append(best_d)

        # Retrain on outer-train with best depth and test on outer-test
        best_model = DecisionTree(criterion=criterion, max_depth=best_d)
        best_model.fit(X_train, y_train)
        y_pred_test = best_model.predict(X_test)
        outer_accs.append(accuracy(y_pred_test, y_test))

    return outer_accs, best_depths

outer_accs, best_depths = evaluate_depth(X, y, depth=None, criterion="information_gain", folds_outer=5, folds_inner=4,
                                         rng=np.random.RandomState(7))

print("=== Q2(b): 5-fold Nested CV ===")
print("Outer-fold accuracies:", [f"{a:.3f}" for a in outer_accs])
print("Mean outer accuracy:", np.mean(outer_accs).round(4))
print("Selected depths per outer fold:", best_depths)

# You should copy these results into: Asst#classification-exp_Q1.md and Asst#classification-exp_Q2.md

# import pandas as pd
# import numpy as np
# import time
# import matplotlib.pyplot as plt
# from tree.base import DecisionTree
# from metrics import *

# np.random.seed(42)
# num_average_time = 100  # Number of times to run each experiment to calculate the average values


# # Function to create fake data (take inspiration from usage.py)
# # ...
# # Function to calculate average time (and std) taken by fit() and predict() for different N and P for 4 different cases of DTs
# # ...
# # Function to plot the results
# # ...
# # Other functions
# # ...
# # Run the functions, Learn the DTs and Show the results/plots



import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from tree.base import DecisionTree

np.random.seed(42)
num_average_time = 10  # run multiple times and average to smooth noise


def make_binary_X(N, M, p=0.5, as_categorical=False):
    X = (np.random.rand(N, M) < p).astype(int)
    df = pd.DataFrame(X, columns=[f"x{j}" for j in range(M)])
    if as_categorical:
        for c in df.columns:
            df[c] = pd.Series(df[c].astype("category"))
    return df


def make_targets(N, case: str):
    """
    case in {"cls", "reg"} to generate discrete or real targets
    """
    if case == "cls":
        # 2-4 classes
        k = np.random.randint(2, 5)
        return pd.Series(np.random.randint(k, size=N))
    else:
        return pd.Series(np.random.randn(N))


def time_fit_predict(X, y, criterion, max_depth=6):
    t0 = time.time()
    dt = DecisionTree(criterion=criterion, max_depth=max_depth)
    dt.fit(X, y)
    t1 = time.time()
    _ = dt.predict(X)
    t2 = time.time()
    return (t1 - t0), (t2 - t1)  # fit_time, predict_time


def avg_time_over_runs(X, y, criterion, runs=num_average_time):
    fit_times, pred_times = [], []
    for _ in range(runs):
        f, p = time_fit_predict(X, y, criterion)
        fit_times.append(f); pred_times.append(p)
    return np.mean(fit_times), np.std(fit_times), np.mean(pred_times), np.std(pred_times)


def experiment_grid(N_list, M_list, case_name):
    """
    case_name in {"realX_clsY","realX_regY","discX_clsY","discX_regY"}
    """
    results = []
    for N in N_list:
        for M in M_list:
            if case_name.startswith("realX"):
                X = pd.DataFrame(np.random.randn(N, M), columns=[f"x{j}" for j in range(M)])
            else:  # discrete X
                as_cat = True  # exercise uses one-hot inside tree
                X = make_binary_X(N, M, p=0.5, as_categorical=as_cat)

            if case_name.endswith("clsY"):
                y = make_targets(N, "cls")
                criterion = "information_gain"
            else:
                y = make_targets(N, "reg")
                criterion = "information_gain"  # ignored for regression split, but needed by ctor

            fmean, fstd, pmean, pstd = avg_time_over_runs(X, y, criterion)
            results.append({
                "N": N, "M": M, "fit_mean": fmean, "fit_std": fstd, "pred_mean": pmean, "pred_std": pstd
            })
            print(f"[{case_name}] N={N}, M={M} -> fit {fmean:.4f}s, pred {pmean:.4f}s")
    return pd.DataFrame(results)


def plot_heatmap(df, value_col, title):
    # Build pivot table N x M for plotting
    pivot = df.pivot(index="N", columns="M", values=value_col)
    plt.figure()
    plt.imshow(pivot.values, aspect="auto")
    plt.title(title)
    plt.xlabel("M (features)")
    plt.ylabel("N (samples)")
    plt.colorbar(label=value_col)
    plt.xticks(range(len(pivot.columns)), pivot.columns)
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.show()


if __name__ == "__main__":
    # Choose a modest grid so runtime is practical
    # N_list = [200, 400, 800, 1600]
    # M_list = [4, 8, 16, 32]
    N_list = [200, 400]
    M_list = [4, 8]

    # Four cases
    df1 = experiment_grid(N_list, M_list, "realX_regY")
    df2 = experiment_grid(N_list, M_list, "realX_clsY")
    df3 = experiment_grid(N_list, M_list, "discX_regY")
    df4 = experiment_grid(N_list, M_list, "discX_clsY")

    # Plot averaged fit & predict times as heatmaps
    for i, df in enumerate([df1, df2, df3, df4], start=1):
        plot_heatmap(df, "fit_mean", f"Case {i}: Fit time (s)")
        plot_heatmap(df, "pred_mean", f"Case {i}: Predict time (s)")

    # Notes for report (put into Asst#experiments_Q1.md):
    # - Empirically, fit time grows roughly O(N * M * log N) due to threshold search on sorted values.
    # - Predict time grows roughly O(depth) per sample; overall O(N * depth).

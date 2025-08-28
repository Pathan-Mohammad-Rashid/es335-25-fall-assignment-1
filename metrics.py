# # # from typing import Union
# # # import pandas as pd


# # # def accuracy(y_hat: pd.Series, y: pd.Series) -> float:
# # #     """
# # #     Function to calculate the accuracy
# # #     """

# # #     """
# # #     The following assert checks if sizes of y_hat and y are equal.
# # #     Students are required to add appropriate assert checks at places to
# # #     ensure that the function does not fail in corner cases.
# # #     """
# # #     assert y_hat.size == y.size
# # #     # TODO: Write here
# # #     pass


# # # def precision(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
# # #     """
# # #     Function to calculate the precision
# # #     """
# # #     pass


# # # def recall(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
# # #     """
# # #     Function to calculate the recall
# # #     """
# # #     pass


# # # def rmse(y_hat: pd.Series, y: pd.Series) -> float:
# # #     """
# # #     Function to calculate the root-mean-squared-error(rmse)
# # #     """

# # #     pass


# # # def mae(y_hat: pd.Series, y: pd.Series) -> float:
# # #     """
# # #     Function to calculate the mean-absolute-error(mae)
# # #     """
# # #     pass



# # from typing import Union
# # import numpy as np
# # import pandas as pd


# # def _assert_same_size(y_hat: pd.Series, y: pd.Series):
# #     assert isinstance(y_hat, pd.Series) and isinstance(y, pd.Series), "Inputs must be pd.Series"
# #     assert y_hat.size == y.size, "y_hat and y must have same size"
# #     assert y.size > 0, "Empty targets"


# # def accuracy(y_hat: pd.Series, y: pd.Series) -> float:
# #     """
# #     Classification accuracy.
# #     """
# #     _assert_same_size(y_hat, y)
# #     return float((y_hat.values == y.values).mean())


# # def precision(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
# #     """
# #     Precision for class 'cls' = TP / (TP + FP)
# #     """
# #     _assert_same_size(y_hat, y)
# #     yh = (y_hat == cls)
# #     yt = (y == cls)
# #     tp = int((yh & yt).sum())
# #     fp = int((yh & ~yt).sum())
# #     denom = tp + fp
# #     return float(tp / denom) if denom > 0 else 0.0


# # def recall(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
# #     """
# #     Recall for class 'cls' = TP / (TP + FN)
# #     """
# #     _assert_same_size(y_hat, y)
# #     yh = (y_hat == cls)
# #     yt = (y == cls)
# #     tp = int((yh & yt).sum())
# #     fn = int((~yh & yt).sum())
# #     denom = tp + fn
# #     return float(tp / denom) if denom > 0 else 0.0


# # def rmse(y_hat: pd.Series, y: pd.Series) -> float:
# #     """
# #     Root-mean-squared-error
# #     """
# #     _assert_same_size(y_hat, y)
# #     diff = y_hat.astype(float).values - y.astype(float).values
# #     return float(np.sqrt(np.mean(diff ** 2)))


# # def mae(y_hat: pd.Series, y: pd.Series) -> float:
# #     """
# #     Mean-absolute-error
# #     """
# #     _assert_same_size(y_hat, y)
# #     diff = y_hat.astype(float).values - y.astype(float).values
# #     return float(np.mean(np.abs(diff)))




# # metrics.py
# from typing import Union
# import pandas as pd
# import numpy as np


# def accuracy(y_hat: pd.Series, y: pd.Series) -> float:
#     """
#     Compute classification accuracy: fraction correct.
#     """
#     assert y_hat.size == y.size
#     if y.size == 0:
#         return 0.0
#     # elementwise equality (works with categorical/string labels)
#     correct = (y_hat == y).sum()
#     return float(correct) / float(y.size)


# def precision(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
#     """
#     Precision for class `cls`: TP / (TP + FP)
#     """
#     assert y_hat.size == y.size
#     # True positives: predicted cls and actual cls
#     tp = ((y_hat == cls) & (y == cls)).sum()
#     # False positives: predicted cls but actual not cls
#     fp = ((y_hat == cls) & (y != cls)).sum()
#     denom = tp + fp
#     if denom == 0:
#         return 0.0
#     return float(tp) / float(denom)


# def recall(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
#     """
#     Recall for class `cls`: TP / (TP + FN)
#     """
#     assert y_hat.size == y.size
#     tp = ((y_hat == cls) & (y == cls)).sum()
#     fn = ((y_hat != cls) & (y == cls)).sum()
#     denom = tp + fn
#     if denom == 0:
#         return 0.0
#     return float(tp) / float(denom)


# def rmse(y_hat: pd.Series, y: pd.Series) -> float:
#     """
#     Root mean squared error (regression).
#     """
#     assert y_hat.size == y.size
#     if y.size == 0:
#         return 0.0
#     # Convert to numeric arrays
#     a = pd.to_numeric(y_hat, errors="coerce").astype(float)
#     b = pd.to_numeric(y, errors="coerce").astype(float)
#     diff = a - b
#     return float(np.sqrt(np.mean(diff ** 2)))


# def mae(y_hat: pd.Series, y: pd.Series) -> float:
#     """
#     Mean absolute error (regression).
#     """
#     assert y_hat.size == y.size
#     if y.size == 0:
#         return 0.0
#     a = pd.to_numeric(y_hat, errors="coerce").astype(float)
#     b = pd.to_numeric(y, errors="coerce").astype(float)
#     return float(np.mean(np.abs(a - b)))





# metrics.py
from typing import Union
import pandas as pd
import numpy as np


def accuracy(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Compute classification accuracy: fraction correct.
    """
    assert y_hat.size == y.size
    if y.size == 0:
        return 0.0
    correct = (y_hat == y).sum()
    return float(correct) / float(y.size)


def precision(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Precision for class `cls`: TP / (TP + FP)
    """
    assert y_hat.size == y.size
    tp = int(((y_hat == cls) & (y == cls)).sum())
    fp = int(((y_hat == cls) & (y != cls)).sum())
    denom = tp + fp
    if denom == 0:
        return 0.0
    return float(tp) / denom


def recall(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Recall for class `cls`: TP / (TP + FN)
    """
    assert y_hat.size == y.size
    tp = int(((y_hat == cls) & (y == cls)).sum())
    fn = int(((y_hat != cls) & (y == cls)).sum())
    denom = tp + fn
    if denom == 0:
        return 0.0
    return float(tp) / denom


def rmse(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Root mean squared error (regression).
    """
    assert y_hat.size == y.size
    if y.size == 0:
        return 0.0
    a = pd.to_numeric(y_hat, errors="coerce").astype(float)
    b = pd.to_numeric(y, errors="coerce").astype(float)
    diff = a - b
    return float(np.sqrt(np.mean(diff ** 2)))


def mae(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Mean absolute error (regression).
    """
    assert y_hat.size == y.size
    if y.size == 0:
        return 0.0
    a = pd.to_numeric(y_hat, errors="coerce").astype(float)
    b = pd.to_numeric(y, errors="coerce").astype(float)
    return float(np.mean(np.abs(a - b)))

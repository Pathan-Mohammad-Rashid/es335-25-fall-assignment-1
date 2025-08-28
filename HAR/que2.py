import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
import seaborn as sns

# Load dataset from MakeDataset.py outputs
# (you may need to adapt this path depending on your folder)
from MakeDataset import X_train, y_train, X_test, y_test

# Combine for EDA
X = np.concatenate((X_train, X_test))
y = np.concatenate((y_train, y_test))

# Activity labels
classes = {1:"WALKING",2:"WALKING_UPSTAIRS",3:"WALKING_DOWNSTAIRS",
           4:"SITTING",5:"STANDING",6:"LAYING"}

def acc_magnitude(sample):
    return np.sqrt(np.sum(sample**2, axis=1))

fig, axes = plt.subplots(6, 1, figsize=(15, 18))
for i, (label, name) in enumerate(classes.items()):
    idx = np.where(y == label)[0][0]
    sample = X[idx]
    mag = acc_magnitude(sample)
    axes[i].plot(mag)
    axes[i].set_title(f"{name} - Magnitude of Acceleration")
plt.tight_layout()
plt.show()

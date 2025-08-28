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

fig, axes = plt.subplots(6, 1, figsize=(15, 18))

for i, (label, name) in enumerate(classes.items()):
    idx = np.where(y == label)[0][0]  # pick first occurrence
    sample = X[idx]  # shape (500,3)
    
    axes[i].plot(sample[:,0], label='accx')
    axes[i].plot(sample[:,1], label='accy')
    axes[i].plot(sample[:,2], label='accz')
    axes[i].set_title(f"{name} - Example waveform")
    axes[i].legend()

plt.tight_layout()
plt.show()

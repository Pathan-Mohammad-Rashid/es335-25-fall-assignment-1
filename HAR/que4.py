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

import tsfel

# Config: choose subset of features (statistical + temporal)
cfg = tsfel.get_features_by_domain()

# Extract features (this will be slow if full dataset)
X_feat = []
for sample in X[:200]:  # use subset for demo (change later)
    df = pd.DataFrame(sample, columns=['accx','accy','accz'])
    feats = tsfel.time_series_features_extractor(cfg, df, verbose=0)
    X_feat.append(feats.values.flatten())

X_feat = np.array(X_feat)

# PCA on TSFEL
pca = PCA(n_components=2)
X_tsfel_pca = pca.fit_transform(X_feat)

plt.figure(figsize=(10,8))
for label, name in classes.items():
    plt.scatter(X_tsfel_pca[y[:200]==label,0], X_tsfel_pca[y[:200]==label,1], label=name, alpha=0.6)
plt.legend()
plt.title("PCA on TSFEL Features")
plt.show()

# Example: correlation for TSFEL features
corr_matrix = pd.DataFrame(X_feat).corr()

plt.figure(figsize=(12,10))
sns.heatmap(corr_matrix, cmap="coolwarm", center=0)
plt.title("Correlation Matrix - TSFEL Features")
plt.show()

# For UCI HAR provided features
corr_matrix_feat = feat_all.corr()

plt.figure(figsize=(12,10))
sns.heatmap(corr_matrix_feat, cmap="coolwarm", center=0)
plt.title("Correlation Matrix - UCI HAR Features")
plt.show()

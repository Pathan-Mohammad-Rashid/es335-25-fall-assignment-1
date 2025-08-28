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


# Flatten samples into vectors (500 timesteps → 500 features)
X_mag = np.array([acc_magnitude(sample) for sample in X])

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_mag)

plt.figure(figsize=(10,8))
for label, name in classes.items():
    plt.scatter(X_pca[y==label,0], X_pca[y==label,1], label=name, alpha=0.6)
plt.legend()
plt.title("PCA on Total Acceleration Magnitude")
plt.show()


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


# Load UCI pre-computed features
train_feat = pd.read_csv("./UCI HAR Dataset/train/X_train.txt", delim_whitespace=True, header=None)
test_feat  = pd.read_csv("./UCI HAR Dataset/test/X_test.txt", delim_whitespace=True, header=None)
feat_all = pd.concat([train_feat, test_feat], axis=0)

labels_train = pd.read_csv("./UCI HAR Dataset/train/y_train.txt", delim_whitespace=True, header=None)[0]
labels_test  = pd.read_csv("./UCI HAR Dataset/test/y_test.txt", delim_whitespace=True, header=None)[0]
labels_all = pd.concat([labels_train, labels_test], axis=0)

# PCA
pca = PCA(n_components=2)
X_dataset_pca = pca.fit_transform(feat_all)

plt.figure(figsize=(10,8))
for label, name in classes.items():
    plt.scatter(X_dataset_pca[labels_all==label,0], X_dataset_pca[labels_all==label,1], label=name, alpha=0.6)
plt.legend()
plt.title("PCA on Provided Dataset Features")
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


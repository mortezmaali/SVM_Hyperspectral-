# -*- coding: utf-8 -*-
"""
Created on Sat May  4 16:39:42 2024

@author: Morteza
"""
import scipy.io as sio
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import accuracy_score

hyperspectral_data = sio.loadmat('C:/Users/Morteza/OneDrive/Desktop/YouTube/coding/Indian_pines.mat')['indian_pines']
hyperspectral_data = hyperspectral_data.astype(np.float64) / 65535.0 

y = sio.loadmat('C:/Users/Morteza/OneDrive/Desktop/YouTube/coding/Indian_pines_gt.mat')['indian_pines_gt']

y = y.reshape((145*145,1))
X = hyperspectral_data.reshape((145*145, 220))

# Number of iterations
n_iterations = 5
accuracy_scores = []
y_pred_full = np.zeros_like(y)

# Splitting data into 5 parts
data_parts = np.array_split(np.random.permutation(range(len(X))), n_iterations)

for i in range(n_iterations):
    test_indices = data_parts[i]
    train_indices = np.concatenate([data_parts[j] for j in range(n_iterations) if j != i])

    # Training and testing data
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    # PCA
    pca = PCA(n_components=100)
    pca.fit(X_train)
    pcX_train = pca.transform(X_train)
    pcX_test = pca.transform(X_test)

    # SVM classifier
    clf = svm.SVC(kernel='poly')
    clf.fit(pcX_train, y_train)

    # Predictions
    y_pred = clf.predict(pcX_test).reshape(-1, 1)
    y_pred_full[test_indices] = y_pred

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_scores.append(accuracy)

# Calculate average accuracy
average_accuracy = sum(accuracy_scores) / n_iterations
print("Average Accuracy:", average_accuracy)

# Reshape predictions and ground truth to the shape of the original image
y_pred_img = y_pred_full.reshape((145, 145))
y_img = y.reshape((145, 145))

# Plot the results
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(y_pred_img, cmap='jet')
plt.title('Predicted')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(y_img, cmap='jet')
plt.title('Ground Truth')
plt.axis('off')

plt.tight_layout()
plt.show()

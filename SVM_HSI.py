# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 20:40:18 2024

@author: Morteza
"""

import scipy.io as sio
import scipy.io
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm


hyperspectral_data = scipy.io.loadmat('C:/Users/Morteza/OneDrive/Desktop/YouTube/coding/Indian_pines.mat')['indian_pines']
hyperspectral_data = hyperspectral_data.astype(np.float64) / 65535.0 

y = scipy.io.loadmat('C:/Users/Morteza/OneDrive/Desktop/YouTube/coding/Indian_pines_gt.mat')['indian_pines_gt']

y= y.reshape((145*145,1))
X= hyperspectral_data.reshape((145*145,220))
pca = PCA(); pcX = pca.fit_transform(X)
ev=pca.explained_variance_ratio_ ; cumulativeVar = np.cumsum(ev)
plt.plot(cumulativeVar)

# Visualize the first 8 principal components
pca = PCA(n_components=8)  # Number of principal components to keep
pca.fit(X)
principal_components = pca.transform(X)

# Reshape each principal component into the dimensions of the hyperspectral image
principal_components_images = []
for i in range(8):
    principal_component_image = principal_components[:, i].reshape(hyperspectral_data.shape[0], hyperspectral_data.shape[1])
    principal_components_images.append(principal_component_image)

# Visualize the principal components as images
plt.figure(figsize=(16, 8))
for i in range(8):
    plt.subplot(2, 4, i + 1)
    plt.imshow(principal_components_images[i], cmap='gray')
    plt.title('Principal Component {}'.format(i + 1))
    plt.axis('off')
plt.tight_layout()
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=11, stratify=y)

nComp = 100
pca = PCA(n_components=nComp)
pca.fit(X_train)
pcX_train = pca.transform(X_train)


clf = svm.SVC(kernel='poly') # Non_Linear Kernel

#Train the model using the training sets
clf.fit(pcX_train, y_train)

pcX_test = pca.transform(X_test)

y_pred = clf.predict(pcX_test)


from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()



from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
import numpy as np
from sklearn import neighbors, datasets
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score

# create color maps for classification problem
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

# initiate dataset
iris = datasets.load_iris()
X = iris.data[:, :2]  
Y = iris.target


# split the data set into training and testing
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.5, random_state=2)

# knn method and fit on training data
knn = neighbors.KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, Y_train)

y_pred = knn.predict(X)

# calculate accuracy
print("Test data accuracy:",np.round(accuracy_score(y_true = Y, y_pred=y_pred),4))

# in sklearn balanced accuracy is presented as mean of sensitivity and specificity
# so we can use this value to get information about data
print("Test data balanced accuracy score:",np.round(balanced_accuracy_score(Y, y_pred),4))


# creating colored map for visualization of classification
# train data map
x_min, x_max = X_train[:, 0].min() - .1, X_train[:, 0].max() + .1
y_min, y_max = X_train[:, 1].min() - .1, X_train[:, 1].max() + .1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),np.linspace(y_min, y_max, 100))
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# test data map
x1_min, x1_max = X_test[:, 0].min() - .1, X_test[:, 0].max() + .1
y1_min, y1_max = X_test[:, 1].min() - .1, X_test[:, 1].max() + .1
xx1, yy1 = np.meshgrid(np.linspace(x1_min, x1_max, 100),np.linspace(y1_min, y1_max, 100))
Z1 = knn.predict(np.c_[xx1.ravel(), yy1.ravel()])
Z1 = Z1.reshape(xx1.shape)


# plot 
# training data
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title('Train data')
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
plt.scatter(X_train[:, 0], X_train[:, 1], c=Y_train, cmap=cmap_bold)
plt.xlabel('sepal length (cm)')
plt.ylabel('sepal width (cm)')
plt.axis('equal')

# test data
plt.subplot(1, 2, 2)
plt.title('Test data')
plt.pcolormesh(xx1, yy1, Z1, cmap=cmap_light)
plt.scatter(X_test[:, 0], X_test[:, 1], c=Y_test, cmap=cmap_bold)
plt.xlabel('sepal length (cm)')
plt.ylabel('sepal width (cm)')
plt.axis('equal')
plt.show()
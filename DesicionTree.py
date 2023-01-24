import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets
from matplotlib.colors import ListedColormap
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Create color maps for 3-class classification problem, as with iris
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

#load iris dataset and take only two first indicators
iris = datasets.load_iris()
X = iris.data[:, :2] 
y = iris.target

#initialization of desicion tree classifier 
clf = DecisionTreeClassifier()
#fit classifier rule
clf.fit(X,y)

#create testing array
x_min, x_max = X[:, 0].min() - .1, X[:, 0].max() + .1
y_min, y_max = X[:, 1].min() - .1, X[:, 1].max() + .1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),np.linspace(y_min, y_max, 100))

#prediction with fitted classifier rule
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

#calculate accuracy score of rule
y_pred = clf.predict(X)
print("Test data accuracy:",accuracy_score(y_true = y, y_pred=y_pred))

#build and show figure (plot)
Z = Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
plt.xlabel('sepal length (cm)')
plt.ylabel('sepal width (cm)')
plt.axis('tight')
plt.show()

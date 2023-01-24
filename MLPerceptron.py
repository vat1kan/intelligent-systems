import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

iris = load_iris()

datasets = train_test_split(iris.data, iris.target,
                            test_size=0.2)

train_data, test_data, train_labels, test_labels = datasets

scaler = StandardScaler()

# fit the train data
scaler.fit(train_data)

# scaling the train data
train_data = scaler.transform(train_data)
test_data = scaler.transform(test_data)

# Analyze the optimal number of node in 1-hidden-layer percepton
nodes = np.arange(1,12)
for i in range(len(nodes)):
    # creating an classifier from the model:
    mlp = MLPClassifier(hidden_layer_sizes=(1, nodes[i]), max_iter=1000)

    # let's fit the training data to our model
    mlp.fit(train_data, train_labels)

    print(f'\nNumber of nodes:{i}\n')
    predictions_train = mlp.predict(train_data)
    print(accuracy_score(predictions_train, train_labels))
    predictions_test = mlp.predict(test_data)
    print(accuracy_score(predictions_test, test_labels))


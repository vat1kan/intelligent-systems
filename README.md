# intelligent-systems
Basic methods of researching data and operations with them

This project implements the basic methods of working with data, their analysis and presentation.

The file "DesicionTree.py" represents the software implementation of the classification method when using a decision tree. 
The prepared array "Fischer's Irises" is used as a data set. An estimate of the classification accuracy is 
calculated and a graph is constructed that displays the test and training samples.

The same process, but using the k-nearest neighbors method, is presented in the file "KNeighbors.py".

The use of neural networks allows for classification with increased accuracy, so their use is increasingly included in human life.
The multilayer perceptron (MLP) is a feedforward artificial neural network model that maps input data sets to a set of appropriate 
outputs. An MLP consists of multiple layers and each layer is fully connected to the following one. The nodes of the layers are 
neurons with  nonlinear activation functions, except for the nodes of the input layer. Between the input and the output layer 
there may be one or more nonlinear hidden layers. The constructed model classifies the data from the "Fischer's Irises" sample. 
Since the task is quite trivial, using one hidden layer is more than enough. A study was carried out at what number of nodes in 
the hidden layer an acceptable accuracy would be achieved.

For convenient work with data, a file "" was also presented, which on an applied task displays the advantage of using various 
methods of classifying and processing data. This problem presents all the passengers of the Titanic and information about 
them: age, gender, whether they survived or not, the class of the cabin where they boarded, and so on. For convenience, Pandas 
is used. This is a fast, powerful, flexible and easy to use open source data analysis and manipulation tool, built on top 
of the Python programming language. This makes it easy to manipulate, classify, sub-sample, and edit the data.
In the course of the work, passengers of what gender and what age category were more likely to be rescued.

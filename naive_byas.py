from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd
import numpy as np

if __name__ == '__main__':
    #Load dataset
    iris = pd.read_csv('datasets/iris.data', header=None)
    print(iris.values.shape)

    feature_names = ['sepal length', 'sepal width', 'petal length', 'petal width']
    target_names = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

    mapping = {'Iris-setosa': 1, 'Iris-versicolor': 2, 'Iris-virginica': 3}
    mplist = []

    for key in mapping:
        mplist.append(key)

    x_values = iris[range(0,len(feature_names))]
    print(x_values.head())

    iris = iris.replace(mapping)
    y_values = iris[[len(feature_names)]]
    
    print('\n------------After mapping------------\n')
    print(y_values.head())

    X_train, X_test, y_train, y_test = train_test_split(x_values.values, y_values.values, test_size=0.3,random_state=109)

    print(y_train.shape)
    y_train = y_train.ravel()
    print(y_train.shape)

    #Create a Gaussian Classifier
    gnb = GaussianNB()

    #Train the model using the training sets
    gnb.fit(X_train, y_train)

    y_pred = gnb.predict(X_test)
    print(y_pred)

    # Model Accuracy, how often is the classifier correct?
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
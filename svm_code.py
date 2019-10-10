import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sn

if __name__ == '__main__':
    iris_data = pd.read_csv('datasets/iris.data', header=None)
    iris_data = iris_data.sample(frac=1) # Get random data

    labels_names = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

    iris_attibutes = iris_data.iloc[:,:-1]
    iris_labels = iris_data.iloc[:,4]

    x_train, x_test, y_train, y_test = train_test_split(iris_attibutes, iris_labels, test_size=0.30)

    model = SVC(kernel='poly', gamma=0.5, degree=10)
    model.fit(x_train, y_train)
    y_predicted = model.predict(x_test)

    print('====================> CONFUSION MATRIX <====================\n')
    confusion_matrix = confusion_matrix(y_test, y_predicted)
    print(confusion_matrix)
    print('============================================================\n')

    print('==================> CLASSIFICATION REPORT <=================\n')
    print(classification_report(y_test, y_predicted, labels=labels_names))
    print('============================================================\n')

    df_cm = pd.DataFrame(confusion_matrix, index = [i for i in labels_names],
                         columns = [i for i in labels_names])
    plt.figure(figsize=(3,3))
    sn.heatmap(df_cm, annot=True, cmap=plt.cm.Blues, square=True)
    plt.yticks(rotation=0)
    plt.show()

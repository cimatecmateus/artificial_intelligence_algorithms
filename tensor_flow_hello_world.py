#!/home/menezes/anaconda3/envs/tensorflow_env/bin/python
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

if __name__ == '__main__':
    data = pd.read_csv('datasets/iris.data', header=None)
    data = data.sample(frac=1) # Get random data
    print(data.head())
    # data.describe()
    # print(data)
    mapping = {'Iris-setosa': 1, 'Iris-versicolor': 2, 'Iris-virginica': 3}
    mplist = []

    for key in mapping:
        mplist.append(key)

    print('\n------------After mapping------------\n')
    data = data.replace(mapping)
    print(data.head())

    print("\nLength of Dataset =", len(data.values))

    dataset_of_trainning = data.iloc[:-50]
    dataset_of_testing = data.iloc[100:]

    print("Length of trainning =", len(dataset_of_trainning.values))
    print("Length of testing =", len(dataset_of_testing.values))

    trainning_input_data = dataset_of_trainning.iloc[:,[0, 1, 2, 3]]
    trainning_label = dataset_of_trainning.iloc[:,[4]]

    testing_input_data = dataset_of_testing.iloc[:,[0, 1, 2, 3]]
    testing_label = dataset_of_testing.iloc[:,[4]]

    # Convert to nparray
    trainning_input_data = trainning_input_data.values
    trainning_label = trainning_label.values
    testing_input_data = testing_input_data.values
    testing_label = testing_label.values

    trainning_label = np.array([[(1 if x == 1 else 0), (1 if x == 2 else 0), (1 if x == 3 else 0)] for x in trainning_label])
    testing_label = np.array([[(1 if x == 1 else 0), (1 if x == 2 else 0), (1 if x == 3 else 0)] for x in testing_label])

    print('\n-------Printing training data-------\n')
    # print(trainning_input_data)
    # print(trainning_label)
    print("trainning_label = ", len(trainning_label))

    print('\n-------Printing testing data-------\n')
    # print(testing_input_data)
    # print(testing_label)
    print("testing_label = ", len(testing_label))

    neural_network_model = tf.keras.models.Sequential()

    neural_network_model.add(tf.keras.layers.Dense(4, input_shape=(len(trainning_input_data[0]),), activation=tf.nn.sigmoid))
    neural_network_model.add(tf.keras.layers.Dense(3, activation=tf.nn.sigmoid))
    neural_network_model.add(tf.keras.layers.Dense(16, activation=tf.nn.tanh))
    neural_network_model.add(tf.keras.layers.Dense(3, activation="sigmoid"))

    lr = 0.01
    neural_network_model.compile(loss='categorical_crossentropy',
                                 optimizer=tf.keras.optimizers.RMSprop(lr=lr),
                                 metrics=['accuracy', 'mse'])

    batch_size = 1
    num_classes = 10
    epochs = 2

    history = neural_network_model.fit(trainning_input_data, trainning_label,
                                       batch_size=batch_size,
                                       epochs=epochs,
                                       verbose=1,
                                       validation_data=(testing_input_data, testing_label))

    loss_value, accuracy_value, mse_value = neural_network_model.evaluate(testing_input_data, testing_label)
    print("Loss value=", loss_value, "Accuracy value =", accuracy_value, "MSE value = ", mse_value)
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR

if __name__ == '__main__':
    dataset_ax = pd.read_csv('unknown_dataset/AX.csv', header=None)
    dataset_ay = pd.read_csv('unknown_dataset/AY.csv', header=None)

    print(dataset_ax.head())
    print(dataset_ay.head())

    print("\nLength of Dataset AX =", len(dataset_ax.values))
    print("Length of Dataset AY =", len(dataset_ay.values))

    print("\nType of Dataset AX =", type(dataset_ax))
    print("Type of Dataset AY =", type(dataset_ay))

    print("\nShape of Dataset AX =", dataset_ax.shape)
    print("Shape of Dataset AY =", dataset_ay.shape)

    ### Mean = 0 significa que é uma sinal periódico centrado em 0
    print("\nDataset AX Signal mean = ", dataset_ax.values.mean(axis=0)) 
    print("Dataset AX Signal max = ", dataset_ax.values.max(axis=0)) 
    print("Dataset AX Signal min = ", dataset_ax.values.min(axis=0))

    ##########################################################################################

    x_train, x_test, y_train, y_test = train_test_split(dataset_ax.values, dataset_ay.values, test_size=0.3)

    print("Length of trainning =", len(x_train))
    print("Length of testing =", len(x_test))

    # Creates two subplots and unpacks the output array immediately
    f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    ax1.set_title('X Train')
    ax1.plot(dataset_ax)
    ax2.set_title('Y Train')
    ax2.plot(dataset_ay)
    plt.show()
    
    neural_network_model = tf.keras.models.Sequential()
    neural_network_model.add(tf.keras.layers.Dense(2, input_shape=(x_train.shape[1],), activation=tf.nn.tanh))
    neural_network_model.add(tf.keras.layers.Dense(3, activation=tf.nn.tanh))
    neural_network_model.add(tf.keras.layers.Dense(16, activation=tf.nn.tanh))
    neural_network_model.add(tf.keras.layers.Dense(y_train.shape[1], activation="relu"))

    lr = 0.01
    neural_network_model.compile(loss='mean_squared_error',
                                 optimizer=tf.keras.optimizers.RMSprop(lr=lr),
                                 metrics=['msle', 'mse'])

    batch_size = 1
    epochs = 30

    history = neural_network_model.fit(x_train, y_train,
                                       batch_size=batch_size,
                                       epochs=epochs,
                                       verbose=1,
                                       validation_data=(x_test, y_test))

    loss_value, msle_value, mse_value = neural_network_model.evaluate(x_test, y_test)
    print("Loss value=", loss_value, "MSLE value =", msle_value, "MSE value = ", mse_value)

    metrics_keys = list(history.history.keys())
    print(metrics_keys)
    # summarize history for accuracy
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)

    # summarize history for loss
    ax1.set_title('Model Metrics')
    ax1.set_ylabel('Loss')
    ax1.set_ylim(0, 1.2*max([max(history.history[metrics_keys[0]]), max(history.history[metrics_keys[3]])]))
    ax1.plot(history.history[metrics_keys[0]], label='train')
    ax1.plot(history.history[metrics_keys[3]], label='test')
    ax1.set_label(ax1.legend(loc='upper right'))

    ax2.set_ylabel('MSLE')
    ax2.set_ylim(0, 1.2*max([max(history.history[metrics_keys[1]]), max(history.history[metrics_keys[4]])])) 
    ax2.plot(history.history[metrics_keys[1]], label='train')
    ax2.plot(history.history[metrics_keys[4]], label='test')
    ax2.set_label(ax2.legend(loc='upper right'))

    ax3.set_ylabel('MSE')
    ax3.set_ylim(0, 1.2*max([max(history.history[metrics_keys[2]]), max(history.history[metrics_keys[5]])])) 
    ax3.plot(history.history[metrics_keys[2]], label='train')
    ax3.plot(history.history[metrics_keys[5]], label='test')
    ax3.set_label(ax3.legend(loc='upper right'))
    ax3.set_xlabel('Epoch')
    
    y_predicted = neural_network_model.predict(x_test)

    f, ax_array = plt.subplots(5, 1, sharex=True)

    f.suptitle("y_test x y_predicted", fontsize=16)

    ax_array[0].set_title('MLP')
    ax_array[0].plot(y_test, 'ro', label='y_test')
    ax_array[0].plot(y_predicted, 'bx', label='y_predicted')
    ax_array[0].set(ylabel="Angle (rad)")
    ax_array[0].legend(loc='upper right')

    # Training classifiers
    svr_array = [SVR(kernel='linear', C=10, gamma='auto'),
                 SVR(kernel='poly', C=20, degree=2),
                 SVR(kernel='rbf', C=15, gamma='auto'),
                 SVR(kernel='sigmoid', gamma='auto')]

    for i, svr in zip(list(range(1, ax_array.shape[0])), svr_array):
        svr.fit(x_train, y_train)
        ax_array[i].set_title(svr.kernel)
        ax_array[i].plot(y_test, 'ro', label='y_test')
        ax_array[i].plot(svr.predict(x_test), 'bx', label='y_predicted')
        ax_array[i].set(ylabel="Angle (rad)")
        if i == (len(ax_array) - 1):
            ax_array[i].set(xlabel="Sample", ylabel="Angle (rad)")
        ax_array[i].legend(loc='upper right')

    plt.show()
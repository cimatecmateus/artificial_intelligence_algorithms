import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR

if __name__ == '__main__':
    dataset_bx = pd.read_csv('unknown_dataset/BX.csv', header=None)
    dataset_by = pd.read_csv('unknown_dataset/BY.csv', header=None)
    
    print("############## Dataset BX head ##############")
    print(dataset_bx.head())
    print("############## Dataset BY head ##############")
    print(dataset_by.head())

    print("\nLength of Dataset BX =", len(dataset_bx.values))
    print("Length of Dataset BX =", len(dataset_by.values))

    print("\nType of Dataset BX =", type(dataset_bx))
    print("Type of Dataset BY =", type(dataset_by))

    print("\nShape of Dataset BX =", dataset_bx.shape)
    print("Shape of Dataset BY =", dataset_by.shape)

    all_as = dataset_bx.iloc[:,0]
    all_bs = dataset_bx.iloc[:,1]
    all_cs = dataset_bx.iloc[:,2]
    
    all_as = all_as.values.tolist()
    all_bs = all_bs.values.tolist()
    all_cs = all_cs.values.tolist()

    x = np.arange(-max(dataset_by.max()), max(dataset_by.max()), 0.1)
    
    for a, b, c, roots in zip(all_as, all_bs, all_cs, dataset_by.values):
        if ((b**2) - 4*a*c) >= 0:
            y = a*(x**2) + b*x + c
            y_ = a*(roots**2) + b*roots + c
            plt.plot(x, y)
            plt.plot(roots, y_, 'ro')

    plt.title('Second Degree Polynomio for each set of coeficients')
    # plt.show()

    x_train, x_test, y_train, y_test = train_test_split(dataset_bx.values, dataset_by.values, test_size=0.3)

    print("Length of trainning =", len(x_train))
    print("Length of testing =", len(x_test))

    neural_network_model = tf.keras.models.Sequential()
    neural_network_model.add(tf.keras.layers.Dense(x_train.shape[1], input_shape=(x_train.shape[1],)))
    # neural_network_model.add(tf.keras.layers.Dense(6))
    # neural_network_model.add(tf.keras.layers.Dense(3))
    neural_network_model.add(tf.keras.layers.Dense(y_train.shape[1]))

    lr = 0.1
    neural_network_model.compile(loss=tf.keras.losses.MeanSquaredLogarithmicError(),
                                 optimizer=tf.keras.optimizers.SGD(lr=lr),
                                 metrics=['msle', 'mse'])

    batch_size = 1
    epochs = 10

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
    plt.show()
    
    num_rows = 5
    y_predicted_mlp = neural_network_model.predict(x_test[range(0,num_rows),:])

    test_as = x_test[range(0,num_rows),0]
    test_bs = x_test[range(0,num_rows),1]
    test_cs = x_test[range(0,num_rows),2]

    x = np.arange(-max(dataset_by.max()), max(dataset_by.max()), 0.1)
    first_iteration = True
    
    for a, b, c, roots in zip(test_as, test_bs, test_cs, y_test):
        if ((b**2) - 4*a*c) >= 0:
            y = a*(x**2) + b*x + c
            y_ = a*(roots**2) + b*roots + c
            plt.plot(x, y)
            if first_iteration:
                plt.plot(roots, y_, 'ro', label='true root')
                first_iteration = False
            else:
                plt.plot(roots, y_, 'ro')
            
    plt.title("y_test x y_predicted", fontsize=16)
    plt.plot(y_predicted_mlp[:,0], 'mx', label='y_predicted_mlp')
    plt.plot(y_predicted_mlp[:,1], 'mx')

    # Training classifiers
    svr_array = [SVR(kernel='linear', C=10, gamma='auto'),
                 SVR(kernel='poly', C=20, degree=2),
                 SVR(kernel='rbf', C=15, gamma='auto'),
                 SVR(kernel='sigmoid', gamma='auto')]

    plt_color_marker = ['rx', 'gx', 'bx', 'yx', 'cx']
    
    for svr, color_marker in zip(svr_array, plt_color_marker):
        svr.fit(x_train, y_train[:,0])
        plt.plot(svr.predict(x_test[range(0,num_rows),:]), np.zeros(svr.predict(x_test[range(0,num_rows),:]).shape), color_marker, label=svr.kernel)
        svr.fit(x_train, y_train[:,1])
        plt.plot(svr.predict(x_test[range(0,num_rows),:]), np.zeros(svr.predict(x_test[range(0,num_rows),:]).shape), color_marker)
        plt.legend(loc='upper right')

    plt.axhline(color='k')
    plt.show()

#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
import random
import math
import logging

class Perceptron():
    def activate(self, perceptron_output):
        return 1 / (1 + math.exp(-perceptron_output))

    def set_random_weights(self, weights_qty):
        self.weights = []    
        for i in range(weights_qty):
            self.weights.append(random.random()*2 - 1)
        logging.debug("Weights = %s", str(self.weights))

    def get_output(self, inputs):
        input_array = np.asarray(inputs)
        weights_array = np.asarray(self.weights)
        return np.dot(input_array, weights_array)

    def add_bias(self, data):
        for i in range(len(data)):
            data[i].append(1.0)
        logging.debug("Data with bias = %s", str(data))
        return data

    def normalize_data(self, data):
        data_array = np.array(data)
        data_normed = data_array / data_array.max()
        logging.debug("Raw data = %s", str(data_array))
        logging.debug("Data normed = %s", str(data_normed))
        return data_normed.tolist()

    def predict(self, input):
        return self.activate(self.get_output(input))

    def set_learning_rate(self, learning_rate):
        self.learning_rate = float(learning_rate)

    def train(self, dataset, right_output, epochs_qty):
        self.dataset = self.normalize_data(dataset)
        self.dataset = self.add_bias(self.dataset)
        self.right_output = right_output
        for i in range(epochs_qty):
            for input_data, right_value in zip(self.dataset, self.right_output):
                network_error = float(right_value) - self.predict(input_data)
                for j in range(len(self.weights)):
                    self.weights[j] += self.learning_rate * network_error * input_data[j]

            print('Epoch: ' + str(i+1))
            print('Weights' + str(self.weights))

    def shows_training_result(self, gradient = False):
        img = np.ones((300, 400)) 
        img_width = img.shape[0]
        img_height = img.shape[1]

        if gradient:
            for x in range(0, img_width):
                for y in range(0, img_height):
                    if (self.predict([float(x)/img_width, float(y)/img_height, 1]) > 0.5):
                        img[x][y] = 1
                    else:
                        img[x][y] = 0 
        else:
            for x in range(0, img_width):
                for y in range(0, img_height):
                    print([float(x)/img_width, float(y)/img_height, 1])
                    print(self.predict([float(x)/img_width, float(y)/img_height, 1]))
                    img[x][y] = self.predict([float(x)/img_width, float(y)/img_height, 1])
        
        plt.imshow(img, origin='lower')
        for x, y in zip(self.dataset, self.right_output):
            print(x[0], x[1])
            if y == 1:
                plt.scatter(x = x[0] * img_height, y = x[1] * img_width, c='r', s=40)
            else:
                plt.scatter(x = x[0] * img_height, y = x[1] * img_width, c='b', s=40)
        # plt.colorbar()
        plt.show()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    data_input_size = 2

    flowers_coordinates = [[1, 10], [1.5, 25], [2, 20], [3, 15], [3, 28], [3.5, 24],
                            [5, 2], [7, 4], [9, 2], [9.5, 12], [10, 4]]

    # 0 - Pink
    # 1 - Violet
    flowers_labels = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

    perceptron = Perceptron()
    perceptron.set_random_weights(3)
    perceptron.set_learning_rate(0.5)
    perceptron.train(flowers_coordinates, flowers_labels, 100)
    perceptron.shows_training_result(gradient=True)

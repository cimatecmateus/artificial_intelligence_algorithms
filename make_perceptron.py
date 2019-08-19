#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
import random
import math
import logging

def activate_perceptron(perceptron_output):
    return 1 / (1 + math.exp(-perceptron_output))

def get_random_weights(weights_qty):
    weights = []    
    for i in range(weights_qty):
        weights.append(random.random()*2 - 1)
    logging.debug("Weights = %s", str(weights))
    return weights

def get_perceptron_output(inputs, weights):
    input_array = np.asarray(inputs)
    weights_array = np.asarray(weights)
    return np.dot(input_array, weights_array)

def add_bias(data):
    data.append(1.0)
    return data

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    data_input_size = 2

    flowers_coordinates = [[1, 10], [1.5, 25], [2, 20], [3,15], [3, 28], [3.5, 24],
                            [5, 2], [7, 4], [9, 2], [9.5, 12], [10, 4]]

    # 0 - Pink
    # 1 - Violet
    flowers_labels = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

    weights_size = 3
    weights = get_random_weights(weights_size)

    img = np.ones((300, 400))
    img_width = img.shape[0]
    img_height = img.shape[1]

    for i in range(0, img_width):
        for j in range(0, img_height):
            pixel = [float(i)/img_width, float(j)/img_height]
            pixel = add_bias(pixel)
            img[i][j] = activate_perceptron(get_perceptron_output(pixel, weights))

            if (img[i][j] > 0.5):
                img[i][j] = 1
            else:
                img[i][j] = 0 

    plt.imshow(img, origin='lower')
    plt.colorbar()
    plt.show() 

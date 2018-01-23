# encoding: utf-8
# python 3

from Model import *
from Layer import DenseLayer
from Activation import Sigmoid
from Optimization import SGD
from Cost.cost_function import cost_derivative

import os
import struct
import numpy as np
from array import array as pyarray
from numpy import array, int8, uint8, zeros
import random


def load_mnist(dataset="training_data", digits=np.arange(10), path="/home/luna/ml/datasets/mnist/"):
    if dataset == "training_data":
        fname_image = os.path.join(path, 'train-images-idx3-ubyte')
        fname_label = os.path.join(path, 'train-labels-idx1-ubyte')
    elif dataset == "testing_data":
        fname_image = os.path.join(path, 't10k-images-idx3-ubyte')
        fname_label = os.path.join(path, 't10k-labels-idx1-ubyte')
    else:
        raise ValueError("dataset must be 'training_data' or 'testing_data'")

    flbl = open(fname_label, 'rb')
    magic_nr, size = struct.unpack(">II", flbl.read(8))
    lbl = pyarray("b", flbl.read())
    flbl.close()

    fimg = open(fname_image, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = pyarray("B", fimg.read())
    fimg.close()

    ind = [k for k in range(size) if lbl[k] in digits]
    N = len(ind)

    images = zeros((N, rows, cols), dtype=uint8)
    labels = zeros((N, 1), dtype=int8)
    for i in range(len(ind)):
        images[i] = array(img[ind[i] * rows * cols: (ind[i] + 1) * rows * cols]).reshape((rows, cols))
        labels[i] = lbl[ind[i]]

    return images, labels


def load_samples(dataset="training_data"):
    image, label = load_mnist(dataset)

    X = [np.reshape(x, (28 * 28, 1)) for x in image]
    X = [x / 255.0 for x in X]  # grey value of (0-255) transform to (0-1)

    # 5 -> [0,0,0,0,0,1.0,0,0,0];  1 -> [0,1.0,0,0,0,0,0,0,0]
    def vectorized_Y(y):
        e = np.zeros((10, 1))
        e[y] = 1.0
        return e

    if dataset == "training_data":
        Y = [vectorized_Y(y) for y in label]
        pair = list(zip(X, Y))
        return pair
    elif dataset == 'testing_data':
        pair = list(zip(X, label))
        return pair
    else:
        print('Something wrong')


if __name__ == '__main__':
    INPUT = 28 * 28
    HIDDEN = 40
    OUTPUT = 10
    sigmoid = Sigmoid()
    sgd = SGD(3.0)
    layer = DenseLayer(INPUT, HIDDEN, sigmoid, "HiddenLayer")
    output_layer = DenseLayer(HIDDEN, OUTPUT, sigmoid, "OutputLayer")
    net = Model(sgd, cost_derivative)
    net.append(layer)
    net.append(output_layer)

    train_set = load_samples(dataset='training_data')
    test_set = load_samples(dataset='testing_data')

    if test_set:
        n_test = len(test_set)

    n = len(train_set)
    for j in range(13):
        random.shuffle(train_set)
        mini_batches = [train_set[k:k + 100] for k in range(0, n, 100)]
        for mini_batch in mini_batches:
            net.train_on_batch(mini_batch)
        if test_set:
            print("Epoch {0}: {1} / {2}".format(j, net.evaluate(test_set), n_test))
        else:
            print("Epoch {0} complete".format(j))

    # 准确率
    correct = 0
    for test_feature in test_set:
        if net.predict(test_feature[0]) == test_feature[1][0]:
            correct += 1
    print("Accuracy: ", correct / len(test_set))
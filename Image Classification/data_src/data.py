import tensorflow as tf
import numpy as np

class MNIST:

    def data_loader(self, path = 'mnist.npz'):

        # Create MNIST dataset with dataloader
        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data(path)

        # Reshape training images to (60,000, 28, 28,1)
        x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
        x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

        # Normalize input images from 0-255 to 0-1
        x_train, x_test = x_train / 255, x_test / 255
        x_val = x_test[7000:]
        y_val = y_test[7000:]
        x_test = x_test[:7000]
        y_test = y_test[:7000]

        return x_train, y_train, x_val, y_val, x_test, y_test


if __name__ == '__main__':
    mnist = MNIST()
    x_train, y_train, x_val, y_val, x_test, y_test = mnist.data_loader('mnist.npz')
    






import os
from tensorflow.examples.tutorials.mnist import input_data


def get_data():
    mnist = input_data.read_data_sets(os.path.join('data', 'MNIST_data', ''), reshape=False, one_hot=False)
    data = {'train_image': mnist.train.images,
            'train_label': mnist.train.labels,
            'test_image': mnist.test.images,
            'test_label': mnist.test.labels}
    return data

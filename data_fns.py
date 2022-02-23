import numpy as np
from sklearn.preprocessing import OneHotEncoder
from tensorflow import keras


def get_mnist_data(expand_x: bool = True):
    (train_x, train_y), (test_x, test_y) = keras.datasets.mnist.load_data()
    if expand_x:
        train_x = np.expand_dims(train_x, axis=-1)
        test_x = np.expand_dims(test_x, axis=-1)
    y_encoder = OneHotEncoder(sparse=False)
    train_y_encoded = y_encoder.fit_transform(np.expand_dims(train_y, axis=1))
    test_y_encoded = y_encoder.fit_transform(np.expand_dims(test_y, axis=1))
    return train_x, train_y_encoded, test_x, test_y_encoded
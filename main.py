import load_canvas
import numpy as np
from tensorflow import keras
from keras.datasets import mnist
from misc_fns import *
from workers.architect import Architect         # from Canvas

(x_train, y_train), (x_test, y_test) = mnist.load_data()
layer_config_list_raw = load_config('mnist_config.json')
layer_config_list = [convert_json_list_tuple(layer) for layer in layer_config_list_raw]

architect = Architect()
model_input_shape = (x_train[0].shape[0], x_train[0].shape[1], 1)
tensor_list = architect.build_model(input_shape=model_input_shape, layer_config_list=layer_config_list)
mnist_model = keras.models.Model(inputs=tensor_list[0], outputs=tensor_list[-1])
mnist_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
mnist_model.fit(x=np.expand_dims(x_train, axis=3), y=y_train, epochs=10)

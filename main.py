import yaml
from tensorflow import keras
from data_fns import get_mnist_data
from Framework.model_builder import ModelBuilder    # added ModelBuilder/Framework to interpreter paths

# get dataset #
train_x, train_y, test_x, test_y = get_mnist_data(expand_x=True)
input_dims = train_x[0].shape
output_dims = train_y[0].__len__()

# get model config from file #
model_config_file = 'configs/conv2d.yml'
with open(model_config_file, 'r') as f:
    yaml_contents = yaml.safe_load(f)
model_config = yaml_contents['Model']

# build model #
model_builder = ModelBuilder()
input_layer, current_layer, build_layer_list = model_builder.build_tensor_model(input_dims=input_dims,
                                                                                model_config=model_config)
output_layer = keras.layers.Dense(output_dims, activation='softmax')(current_layer)
model = keras.models.Model(input_layer, output_layer)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# train model #
model.fit(train_x, train_y)
model.evaluate(test_x, test_y)
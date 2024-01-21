from keras.models import load_model
from keras import layers
import tensorflow as tf
from matplotlib import pyplot
from keras.applications.convnext import preprocess_input
import numpy as np


# Added classes (ConvNeXtTiny cannot be loaded from memory unless we do this)

class StochasticDepth(layers.Layer):
    """Stochastic Depth module.

    It performs batch-wise dropping rather than sample-wise. In libraries like
    `timm`, it's similar to `DropPath` layers that drops residual paths
    sample-wise.

    References:
      - https://github.com/rwightman/pytorch-image-models

    Args:
      drop_path_rate (float): Probability of dropping paths. Should be within
        [0, 1].

    Returns:
      Tensor either with the residual path dropped or kept.
    """

    def __init__(self, drop_path_rate, **kwargs):
        super().__init__(**kwargs)
        self.drop_path_rate = drop_path_rate

    def call(self, x, training=None):
        if training:
            keep_prob = 1 - self.drop_path_rate
            shape = (tf.shape(x)[0],) + (1,) * (len(tf.shape(x)) - 1)
            random_tensor = keep_prob + tf.random.uniform(shape, 0, 1)
            random_tensor = tf.floor(random_tensor)
            return (x / keep_prob) * random_tensor
        return x

    def get_config(self):
        config = super().get_config()
        config.update({"drop_path_rate": self.drop_path_rate})
        return config


class LayerScale(layers.Layer):
    """Layer scale module.

    References:
      - https://arxiv.org/abs/2103.17239

    Args:
      init_values (float): Initial value for layer scale. Should be within
        [0, 1].
      projection_dim (int): Projection dimensionality.

    Returns:
      Tensor multiplied to the scale.
    """

    def __init__(self, init_values, projection_dim, **kwargs):
        super().__init__(**kwargs)
        self.init_values = init_values
        self.projection_dim = projection_dim

    def build(self, input_shape):
        self.gamma = tf.Variable(
            self.init_values * tf.ones((self.projection_dim,))
        )

    def call(self, x):
        return x * self.gamma

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "init_values": self.init_values,
                "projection_dim": self.projection_dim,
            }
        )
        return config

# Load best model
model = load_model('model-best.h5', compile=False, custom_objects={
    "StochasticDepth": StochasticDepth,
    "LayerScale": LayerScale,
})

for i, layer in enumerate(model.layers):
    print(i, layer.name)
# Get one image from the test set

layer_dict = {}
for i, layer in enumerate(model.layers):
    if 'depthwise_conv' not in layer.name:
        continue
    print(i, layer.name, layer.output.shape)

n_layer = 142
filters, biases = model.layers[n_layer].get_weights()
# normalize filter values to 0-1 so we can visualize them
f_min, f_max = filters.min(), filters.max()
filters = (filters - f_min) / (f_max - f_min)
# plot first few filters
n_filters, ix = 25, 1
for i in range(n_filters):
    # get the filter
    f = filters[:, :, :, i]
    # plot each channel separately
    # specify subplot and turn of axis
    ax = pyplot.subplot(n_filters, 5, ix)
    ax.set_xticks([])
    ax.set_yticks([])
    # plot filter channel in grayscale
    pyplot.imshow(f[:, :], cmap='gray')
    ix += 1
# show the figure
pyplot.show()

feature_model = tf.keras.Model(inputs=model.inputs, outputs=model.layers[n_layer].output)
img = tf.keras.preprocessing.image.load_img('./MIT_small_train_1/test/Opencountry/open53.jpg',
                                            target_size=(224, 224))

# Convert to numpy array
img = tf.keras.preprocessing.image.img_to_array(img)
# Expand the dimensions to be (1, 224, 224, 3)
img = np.expand_dims(img, axis=0)
# Preprocess the image (normalize the image)
img = preprocess_input(img)

# Get the features
feature_maps = feature_model.predict(img)

# Plot the feature maps
square = 9
ix = 1
for _ in range(square):
     for _ in range(square):
         # specify subplot and turn of axis
         ax = pyplot.subplot(square, square, ix)
         ax.set_xticks([])
         ax.set_yticks([])
         # plot filter channel in grayscale
         pyplot.imshow(feature_maps[0, :, :, ix-1], cmap='gray')
         ix += 1
# show the figure
pyplot.show()









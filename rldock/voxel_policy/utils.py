import numpy as np
import tensorflow as tf
# import tensorflow.nn
# from stable_baselines.a2c.utils import linear, conv_to_fc
#
#
# def ortho_init(scale=1.0):
#     """
#     Orthogonal initialization for the policy weights
#
#     :param scale: (float) Scaling factor for the weights.
#     :return: (function) an initialization function for the weights
#     """
#
#     # _ortho_init(shape, dtype, partition_info=None)
#     def _ortho_init(shape, *_, **_kwargs):
#         """Intialize weights as Orthogonal matrix.
#
#         Orthogonal matrix initialization [1]_. For n-dimensional shapes where
#         n > 2, the n-1 trailing axes are flattened. For convolutional layers, this
#         corresponds to the fan-in, so this makes the initialization usable for
#         both dense and convolutional layers.
#
#         References
#         ----------
#         .. [1] Saxe, Andrew M., James L. McClelland, and Surya Ganguli.
#                "Exact solutions to the nonlinear dynamics of learning in deep
#                linear
#         """
#         # lasagne ortho init for tf
#         shape = tuple(shape)
#         if len(shape) == 2:
#             flat_shape = shape
#         elif len(shape) == 4:  # assumes NHWC
#             flat_shape = (np.prod(shape[:-1]), shape[-1])
#         elif len(shape) == 5:  # assumes NHWDC
#             flat_shape = (np.prod(shape[:-1]), shape[-1])
#         else:
#             raise NotImplementedError
#         gaussian_noise = np.random.normal(0.0, 1.0, flat_shape)
#         u, _, v = np.linalg.svd(gaussian_noise, full_matrices=False)
#         weights = u if u.shape == flat_shape else v  # pick the one with the correct shape
#         weights = weights.reshape(shape)
#         return (scale * weights[:shape[0], :shape[1]]).astype(np.float32)
#
#     return _ortho_init
#
#
# def voxel_conv(input_tensor, scope, *, n_filters, filter_size, stride,
#                pad='VALID', init_scale=1.0, data_format='NDHWC', one_dim_bias=False):
#     """
#     Creates a 3d convolutional layer for TensorFlow
#
#     :param input_tensor: (TensorFlow Tensor) The input tensor for the convolution
#     :param scope: (str) The TensorFlow variable scope
#     :param n_filters: (int) The number of filters
#     :param filter_size:  (Union[int, [int], tuple<int, int>]) The filter size for the squared kernel matrix,
#     or the height and width of kernel filter if the input is a list or tuple
#     :param stride: (int) The stride of the convolution
#     :param pad: (str) The padding type ('VALID' or 'SAME')
#     :param init_scale: (int) The initialization scale
#     :param data_format: (str) The data format for the convolution weights
#     :param one_dim_bias: (bool) If the bias should be one dimentional or not
#     :return: (TensorFlow Tensor) 3d convolutional layer
#     """
#     if isinstance(filter_size, list) or isinstance(filter_size, tuple):
#         assert len(filter_size) == 3, \
#             "Filter size must have 2 elements (height, width), {} were given".format(len(filter_size))
#         filter_height = filter_size[0]
#         filter_width = filter_size[1]
#         filter_z = filter_size[2]
#     else:
#         filter_height = filter_size
#         filter_width = filter_size
#         filter_z = filter_size
#     if data_format == 'NDHWC':
#         channel_ax = 4
#         strides = [1, stride, stride, stride, 1]
#         bshape = [1, 1, 1, 1, n_filters]
#     elif data_format == 'NCDHW':
#         channel_ax = 1
#         strides = [1, 1, stride, stride, stride]
#         bshape = [1, n_filters, 1, 1, 1]
#     else:
#         raise NotImplementedError
#     bias_var_shape = [n_filters] if one_dim_bias else [1, 1, 1, 1, n_filters]
#     n_input = input_tensor.get_shape()[channel_ax].value
#     wshape = [filter_z, filter_height, filter_width, n_input, n_filters]
#     with tf.variable_scope(scope):
#         weight = tf.get_variable("w", wshape, initializer=ortho_init(init_scale))
#         bias = tf.get_variable("b", bias_var_shape, initializer=tf.constant_initializer(0.0))
#         if not one_dim_bias and data_format == 'NHWC':
#             bias = tf.reshape(bias, bshape)
#         return bias + tf.nn.conv3d(input_tensor, weight, strides=strides, padding=pad, data_format=data_format)
#
#
# def voxel_nature_cnn(scaled_images, **kwargs):
#     """
#     CNN from Nature paper.
#
#     :param scaled_images: (TensorFlow Tensor) Image input placeholder
#     :param kwargs: (dict) Extra keywords parameters for the convolutional layers of the CNN
#     :return: (TensorFlow Tensor) The CNN output layer
#     """
#     activ = tf.nn.relu
#     layer_1 = activ(voxel_conv(scaled_images, 'c1', n_filters=8, filter_size=1, stride=1, init_scale=np.sqrt(2), **kwargs))
#     layer_2 = activ(voxel_conv(layer_1, 'c2', n_filters=64, filter_size=6, stride=2, init_scale=np.sqrt(2), **kwargs))
#     layer_3 = activ(voxel_conv(layer_2, 'c3', n_filters=64, filter_size=3, stride=2, init_scale=np.sqrt(2), **kwargs))
#     layer_4 = conv_to_fc(layer_3)
#     return activ(linear(layer_4, 'fc1', n_hidden=512, init_scale=np.sqrt(2)))


# def squeeze_module(image, sx, ex1, ex2, p, sc, ec):
#     layer_1 = tf.nn.relu(voxel_conv(image, 's1' + str(p), n_filters=sc, filter_size=sx, pad='SAME', stride=1))
#     layer_2 = tf.nn.relu(voxel_conv(layer_1, 's2' + str(p), n_filters=ec, filter_size=ex1, pad='SAME', stride=1))
#     layer_3 = tf.nn.relu(voxel_conv(layer_1, 's3' + str(p), n_filters=ec, filter_size=ex2, pad='SAME', stride=1))
#
#     return tf.concat([layer_2, layer_3], axis=-1)

import tensorflow.keras as k
def keras_squeeze_module(incoming_layer, sx, ex1, ex2, p, sc, ec):
    layer_1 = k.layers.Conv3D(sc, sx,  padding='SAME', strides=1, activation='relu', name="s1" + str(p))(incoming_layer)
    layer_2 = k.layers.Conv3D(ec, ex1, padding='SAME', strides=1, activation='relu', name="s2" + str(p))(layer_1)
    layer_3 = k.layers.Conv3D(ec, ex2, padding='SAME', strides=1, activation='relu', name="s3" + str(p))(layer_1)

    return k.layers.Concatenate(name='hi' + str(p))([layer_2, layer_3])

def kerasVoxelExtractor(im):
    layer_1 = k.layers.Conv3D(96, 1, strides=2, activation='relu',name='convfirst')(im)
    layer_2 = keras_squeeze_module(layer_1, 1, 1, 3, 1, 16, 64)
    ll = k.layers.BatchNormalization()(layer_2)
    layer_3 = keras_squeeze_module(ll, 1, 1, 3, 2, 16, 64)
    layer_4 = keras_squeeze_module(layer_3, 1, 1, 3, 3, 32, 128)
    layer_5 = k.layers.MaxPool3D(2, 2, 'VALID',name='maxps')(layer_4)
    layer_6 = keras_squeeze_module(layer_5, 1, 1, 3, 4, 32, 128)
    layer_7 = keras_squeeze_module(layer_6, 1, 1, 3, 5, 48, 192)
    ll = k.layers.BatchNormalization()(layer_7)
    # layer_8 = keras_squeeze_module(ll, 1, 1, 3, 6, 48, 192)
    # layer_9 = keras_squeeze_module(layer_8, 1, 1, 3, 7, 64, 256)
    layer_10 = k.layers.AveragePooling3D(2, 2, 'VALID', name='avgps')(ll)
    return k.layers.Flatten(name='flat')(layer_10)

# '''
# https://pubs.acs.org/doi/10.1021/acs.jcim.7b00650
# '''
# def voxel_kdeep_paper(im, **kwargs):
#     layer_1 = tf.nn.relu(voxel_conv(im, 'c1', n_filters=96, filter_size=1, stride=2, init_scale=np.sqrt(2), **kwargs))
#     layer_2 = tf.nn.relu(squeeze_module(layer_1, 1, 1, 3, 1, 16, 64))
#     layer_3 = tf.nn.relu(squeeze_module(layer_2, 1, 1, 3, 2, 16, 64))
#     layer_4 = tf.nn.relu(squeeze_module(layer_3, 1, 1, 3, 3, 32, 128))
#     layer_5 = tf.nn.max_pool3d(layer_4, 2, 2, 'VALID')
#     layer_6 = tf.nn.relu(squeeze_module(layer_5, 1, 1, 3, 4, 32, 128))
#     layer_7 = tf.nn.relu(squeeze_module(layer_6, 1, 1, 3, 5, 48, 192))
#     layer_8 = tf.nn.relu(squeeze_module(layer_7, 1, 1, 3, 6, 48, 192))
#     layer_9 = tf.nn.relu(squeeze_module(layer_8, 1, 1, 3, 7, 64, 256))
#     layer_10 = tf.nn.avg_pool3d(layer_9, 2, 2, 'VALID')
#     return conv_to_fc(layer_10)

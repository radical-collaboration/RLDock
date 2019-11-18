import tensorflow as tf
import tensorflow.keras as k

def lrelu(x):
    return tf.keras.activations.relu(x, alpha=0.1)

def keras_squeeze_module(incoming_layer, sx, ex1, ex2, p, sc, ec):
    layer_1 = k.layers.Conv3D(sc, sx,  padding='SAME', strides=1, activation=lrelu, name="s1" + str(p))(incoming_layer)
    layer_2 = k.layers.Conv3D(ec, ex1, padding='SAME', strides=1, activation=lrelu, name="s2" + str(p))(layer_1)
    # layer_3 = k.layers.Conv3D(ec, ex2, padding='SAME', strides=1, activation=lrelu, name="s3" + str(p))(layer_1)
    return layer_2

def kerasVoxelExtractor(im):
    layer_1 = k.layers.Conv3D(64, 3, strides=2, activation=lrelu,name='convfirst')(im)
    layer_2 = keras_squeeze_module(layer_1, 1, 3, 3, 1, 16, 64)
    ll = k.layers.BatchNormalization(name='bbn0')(layer_2)
    layer_3 = keras_squeeze_module(ll, 1, 3, 3, 2, 16, 64)
    layer_4 = keras_squeeze_module(layer_3, 1, 3, 3, 3, 32, 96)
    ll = k.layers.BatchNormalization(name='bbn0.2')(layer_4)
    layer_5 = k.layers.MaxPool3D(2, 2, 'VALID',name='maxps')(ll)
    layer_6 = keras_squeeze_module(layer_5, 1, 3, 3, 4, 32, 128)
    layer_7 = keras_squeeze_module(layer_6, 1, 3, 3, 5, 48, 128)
    ll = k.layers.BatchNormalization(name='bbn1')(layer_7)
    layer_8 = keras_squeeze_module(ll, 1, 3, 3, 6, 48, 192)
    layer_9 = keras_squeeze_module(layer_8, 1, 3, 3, 7, 64, 128)
    ll = k.layers.BatchNormalization(name='bbn1.2')(layer_9)
    layer_10 = k.layers.AveragePooling3D(2, 2, 'VALID', name='avgps')(ll)
    return k.layers.Flatten(name='flat')(layer_10)

import numpy as np
import tensorflow as tf

act_funcs = {'elu': tf.nn.elu,
             'sigmoid': tf.nn.sigmoid,
             'tanh': tf.nn.tanh}


def conv_2d(input_name, shape, params, size, layer_id, train=True):
    x = tf.keras.Input(shape=tuple(shape), dtype=tf.float32, name=input_name)
    x_shape1 = x.get_shape().as_list()
    if len(x_shape1) == 3:
        x = tf.expand_dims(x, axis=-1)
        x_shape1 = x.get_shape().as_list()
    x_shape1 = np.array(x_shape1[1:3])
    x_shape = x_shape1 // size
    x_shape[x_shape <= 1] = 2
    x_shape = np.minimum(x_shape, x_shape1)
    kernels = x_shape.tolist()
    act_func = act_funcs[params['act_func']]
    x_shape = x_shape // 2
    x_shape[x_shape == 0] = 1
    pool_size = x_shape.tolist()
    pool_size = [int(p) for p in pool_size]
    kernels = [int(k) for k in kernels]
    name = 'conv_2d_' + layer_id
    conv = tf.keras.layers.Conv2D(filters=int(params['filters']),
                                  kernel_size=kernels,
                                  padding="valid",
                                  name='conv_2d_' + layer_id,
                                  activation=act_func)(x)

    pool = tf.keras.layers.AveragePooling2D(pool_size=pool_size, strides=1, name='pool' + layer_id)(conv)

    cnn_model = tf.keras.Model(x, pool, name='model_' + name)

    print(f"layer {name} has shape {pool.get_shape().as_list()}")
    print(f"layer weights {name} has shape {cnn_model.trainable_weights[0].get_shape().as_list()}")
    return cnn_model


def conv_3d(input_name, shape, params, size, layer_id, train=True):
    x = tf.keras.Input(shape=tuple(shape), dtype=tf.float32, name=input_name)
    x_shape1 = x.get_shape().as_list()
    if len(x_shape1) == 3:
        x = tf.expand_dims(x, axis=-1)
        x_shape1 = x.get_shape().as_list()
    x_shape1 = np.array(x_shape1[1:4])
    x_shape = x_shape1 // size
    x_shape[x_shape <= 1] = 2
    x_shape = np.minimum(x_shape, x_shape1)
    kernels = x_shape.tolist()
    if len(kernels) != 3:
        kernels = [1] + kernels

    act_func = act_funcs[params['act_func']]
    x_shape = x_shape // 2
    x_shape[x_shape == 0] = 1
    x_shape = x_shape // 2
    x_shape[x_shape == 0] = 1
    pool_size = x_shape.tolist()
    pool_size = [int(p) for p in pool_size]
    kernels = [int(k) for k in kernels]
    name = 'conv_3d_' + layer_id
    conv = tf.keras.layers.Conv3D(filters=int(params['filters']),
                                  kernel_size=kernels,
                                  padding="valid",
                                  name='conv_3d_' + layer_id,
                                  activation=act_func)

    pool = tf.keras.layers.AveragePooling3D(pool_size=pool_size, strides=1, name='pool' + layer_id)
    if len(x.get_shape().as_list()) == 4:
        cnn_output = conv(tf.expand_dims(x, axis=-1))
        cnn_output = pool(cnn_output)
    else:
        cnn_output = conv(x)
        cnn_output = pool(cnn_output)
    cnn_model = tf.keras.Model(x, cnn_output, name='model_' + name)
    print(f"layer {name} has shape {cnn_output.get_shape().as_list()}")
    print(f"layer weights {name} has shape {cnn_model.trainable_weights[0].get_shape().as_list()}")
    return cnn_model


def time_distr_conv_2d(input_name, shape, params, size, layer_id, train=True):
    x = tf.keras.Input(shape=tuple(shape), dtype=tf.float32, name=input_name)
    x_shape1 = x.get_shape().as_list()
    if len(x_shape1) == 3:
        x = tf.expand_dims(x, axis=-1)
        x_shape1 = x.get_shape().as_list()
    if len(x_shape1) == 4:
        x = tf.expand_dims(x, axis=-1)
    x_shape1 = np.array(x_shape1[2:4])
    x_shape = x_shape1 // size
    x_shape[x_shape <= 1] = 2
    x_shape = np.minimum(x_shape, x_shape1)
    kernels = x_shape.tolist()
    act_func = act_funcs[params['act_func']]
    x_shape = x_shape // 2
    x_shape[x_shape == 0] = 1
    pool_size = x_shape.tolist()
    pool_size = [int(p) for p in pool_size]
    kernels = [int(k) for k in kernels]
    name = 'time_distr_conv_2d_' + layer_id
    conv = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(filters=int(params['filters']),
                                                                  kernel_size=kernels,
                                                                  padding="same",
                                                                  name='conv_' + layer_id,
                                                                  activation=act_func),
                                           name='time_distr_conv_2d_' + layer_id)(x)

    pool = tf.keras.layers.TimeDistributed(tf.keras.layers.AveragePooling2D(pool_size=pool_size,
                                                                            strides=1,
                                                                            name='pool' + layer_id),
                                           name='time_distr_pool_2d_' + layer_id)(conv)
    cnn_model = tf.keras.Model(x, pool, name='model_' + name)

    print(f"layer {name} has shape {pool.get_shape().as_list()}")
    print(f"layer weights {name} has shape {cnn_model.trainable_weights[0].get_shape().as_list()}")
    return cnn_model


def time_distr_conv_3d(input_name, shape, params, size, layer_id, train=True):
    x = tf.keras.Input(shape=tuple(shape), dtype=tf.float32, name=input_name)
    x_shape1 = x.get_shape().as_list()
    if len(x_shape1) == 3:
        x = tf.expand_dims(x, axis=-1)
        x_shape1 = x.get_shape().as_list()
    if len(x_shape1) == 4:
        x = tf.expand_dims(x, axis=-1)
    x_shape1 = np.array(x_shape1[2:5])
    x_shape = x_shape1 // size
    x_shape[x_shape <= 1] = 2
    x_shape = np.minimum(x_shape, x_shape1)
    kernels = x_shape.tolist()
    act_func = act_funcs[params['act_func']]
    x_shape = x_shape // 2
    x_shape[x_shape == 0] = 1
    pool_size = x_shape.tolist()
    pool_size = [int(p) for p in pool_size]
    kernels = [int(k) for k in kernels]
    name = 'time_distr_conv_3d_' + layer_id
    conv = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv3D(filters=int(params['filters']),
                                                                  kernel_size=kernels,
                                                                  padding="same",
                                                                  name='conv_' + layer_id,
                                                                  activation=act_func),
                                           name='time_distr_conv_3d_' + layer_id)

    pool = tf.keras.layers.TimeDistributed(tf.keras.layers.AveragePooling3D(pool_size=pool_size,
                                                                            strides=1,
                                                                            name='pool' + layer_id),
                                           'time_distr_pool_3d_' + layer_id)
    if len(x.get_shape().as_list()) == 5:
        cnn_output = conv(tf.expand_dims(x, axis=-1))
        cnn_output = pool(cnn_output)
    else:
        cnn_output = conv(x)
        cnn_output = pool(cnn_output)
    cnn_model = tf.keras.Model(x, cnn_output, name='model_' + name)
    print(f"layer {name} has shape {cnn_output.get_shape().as_list()}")
    print(f"layer weights {name} has shape {cnn_model.trainable_weights[0].get_shape().as_list()}")
    return cnn_model


def lstm(input_name, shape, params, size, layer_id, train=False):
    x = tf.keras.Input(shape=tuple(shape), dtype=tf.float32, name=input_name)
    # if train:
    #     tf.compat.v1.disable_eager_execution()
    #     tf.compat.v1.experimental.output_all_intermediates(True)
    x_shape = x.get_shape().as_list()
    if len(x_shape) != 3:
        x = tf.reshape(x, [-1, x_shape[1], np.prod(x_shape[2:])])
    name = 'lstm_' + layer_id
    lstm = tf.keras.layers.LSTM(
        int(size * x_shape[1]),
        activation='tanh',
        recurrent_activation='sigmoid',
        name=name,
        return_sequences=True,
        unroll=False,
        use_bias=True,
        recurrent_dropout=0)(x)
    lstm_model = tf.keras.Model(x, lstm, name='model_' + name)
    print(f"layer {name} has shape {lstm.get_shape().as_list()}")
    print(f"layer weights {name} has shape {lstm_model.trainable_weights[0].get_shape().as_list()}")

    return lstm_model


def dense(input_name, shape, params, size, layer_id, train=True):
    if params['act_func'] is not None:
        act_func = act_funcs[params['act_func']]
    else:
        act_func = None
    name = 'dense_' + layer_id
    x = tf.keras.Input(shape=tuple(shape), dtype=tf.float32, name=input_name)
    out_dense = tf.keras.layers.Dense(units=size, activation=act_func, name=name)(x)
    dense_model = tf.keras.Model(x, out_dense, name='_'.join(['model', name]))
    print(f"layer {name} has shape {out_dense.get_shape().as_list()}")
    print(f"layer weights {name} has shape {dense_model.trainable_weights[0].get_shape().as_list()}")
    return dense_model


def layers_func():
    layers = {'conv_2d': conv_2d,
              'time_distr_conv_2d': time_distr_conv_2d,
              'conv_3d': conv_3d,
              'time_distr_conv_3d': time_distr_conv_3d,
              'lstm': lstm,
              'dense': dense
              }
    return layers

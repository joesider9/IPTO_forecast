import copy

import numpy as np
import tensorflow as tf

from eforecast.deep_models.tf_2x.layers import layers_func
from eforecast.deep_models.tf_2x.global_builders import RBFLayer
from eforecast.deep_models.tf_2x.global_builders import apply_activations
from eforecast.deep_models.tf_2x.global_builders import act_nan_layer

layers_functions = layers_func()


def get_size(size, layer_id, layers, output_shape):
    if isinstance(size, set):
        size = list(size)
    if isinstance(size, list):
        size = size[0]
    if size == 'linear':
        if layer_id + 1 >= len(layers):
            raise ValueError('Cannot be linear the last layer')
        size_next = layers[layer_id + 1][1]
        if isinstance(size_next, set):
            size_next = list(size_next)
        if isinstance(size_next, list):
            size_next = size_next[0]
        return int((size_next + np.prod(output_shape[1:])) / 2)
    elif size < 8:
        return int(size * np.prod(output_shape))
    else:
        return size


def build_layer(input_name, shape, layers, name_scope, params, train=True, is_for_cluster=False):
    input_layer = tf.keras.Input(shape=shape, dtype=tf.float32, name=input_name)
    output_layer = input_layer
    output_shape = shape
    with tf.name_scope(name_scope) as scope:
        print(f'Graph of {name_scope} building')
        layers_built = dict()
        for layer_id, layer_tuple in enumerate(layers):
            layer_name, size = layer_tuple
            print(f'Input has shape {output_shape}')
            if len(output_shape) == 3 and '3d' in layer_name:
                layer_tuple = ('conv_2d', size)
                layer_name, size = layer_tuple
            if layer_name == 'dense' and not is_for_cluster:
                size = get_size(size, layer_id, layers, output_shape)
            if isinstance(size, set):
                size = list(size)
            if isinstance(size, list):
                if len(size) > 0:
                    size = size[0]

            if layer_name not in {'Flatten', 'Dropout', 'Reshape'}:
                if layer_name == 'lstm':
                    lstm_lags = output_layer.get_shape()[1]
                layer_ = layers_functions[layer_name](input_name, output_shape, params, size,
                                                      str(layer_id), train=train)
                layers_built[layer_.name] = layer_
            elif layer_name == 'Reshape':
                layer_ = tf.keras.layers.Reshape([lstm_lags,
                                                       int(output_layer.get_shape()[1].value / lstm_lags)],
                                                 name=f'Reshape_{layer_id}')
                layers_built[layer_.name] = layer_
            elif layer_name == 'Dropout':
                layer_ = tf.keras.layers.Dropout(size)
                layers_built[layer_.name] = layer_
            elif layer_name == 'Flatten':
                layer_ = tf.keras.layers.Flatten()
                layers_built[layer_.name] = layer_
            else:
                names = [name for name in layers_functions.keys()]
                raise ValueError(f"Unknown layer name {layer_name}. Valid names {names}")
            output_layer = layer_(output_layer)
            output_shape = output_layer.get_shape().as_list()[1:]
            input_name = layer_.output.name


    model_layer = tf.keras.Model(input_layer, output_layer, name='model_prime_' + name_scope)
    return model_layer, layers_built


def proba_output(shape, quantiles):
    x = tf.keras.Input(tuple(shape[1:]), name='model_output')
    output_layers = []
    outputs = []
    for i, q in enumerate(quantiles):
        # Get output layers
        layer_ = tf.keras.layers.Dense(units=1,
                                       name="{}_q{}".format(i, int(q * 100)))
        outputs.append(layer_(x))
        output_layers.append(layer_)
    proba_out = tf.keras.Model(x, outputs)
    return proba_out, output_layers


def build_graph(x, model_layers, params, is_fuzzy=False, is_global=False, is_for_cluster=False, train=True,
                probabilistic=False, thres_act=None, quantiles=None):
    inputs = dict()
    shapes = dict()
    if isinstance(x, dict):
        for name, inp in sorted(x.items()):
            if 'act' not in name and 'clustering' not in name:
                shape = tuple(inp.shape[1:])
                inp_name = 'input_' + name if name != 'input' else 'input'
                if 'data_row' in x.keys():
                    shape_row = tuple(x['data_row'].shape[1:])

                    inputs[inp_name] = [tf.keras.Input(shape, dtype=tf.float32, name=inp_name),
                                        tf.keras.Input(shape_row, dtype=tf.float32, name='data_row')]
                    shapes[inp_name] = [shape, shape_row]
                else:
                    inputs[inp_name] = tf.keras.Input(shape, dtype=tf.float32, name='input_' + name)
                    shapes[inp_name] = shape
    else:
        shape = tuple(x.shape[1:])
        inputs['input'] = tf.keras.Input(shape, dtype=tf.float32, name='input')
        shapes['input'] = shape

    model_output_dict = dict()
    name_scope_list = []
    model_layers_built = dict()
    for name_scope in sorted(inputs.keys()):
        input_name = name_scope
        shape = shapes[name_scope] if not isinstance(inputs, list) else shapes[name_scope][0]
        layer = model_layers['input']
        params_temp = copy.deepcopy(params)

        model_layer, prime_layers = build_layer(input_name, shape,
                                                layer,
                                                f'prime_{name_scope}',
                                                params_temp,
                                                train=train, is_for_cluster=is_for_cluster)
        if 'data_row' in x.keys():
            with tf.name_scope(f'{name_scope}_data_row') as scope:
                shape = shapes[name_scope][1]
                model_layer_row, row_layers = build_layer('data_row', shape, model_layers['data_row'],
                                                          f'prime_{name_scope}_data_row', params,
                                                          train=train)
                prime_layers.update(row_layers)
        else:
            model_layer_row = None
        if model_layer_row is not None:
            output_layer = tf.keras.layers.concatenate([model_layer(inputs[name_scope][0]),
                                                        model_layer_row(inputs[name_scope][1])], 1,
                                                       name=f'concat_{name_scope}_row')
        else:
            output_layer = model_layer(inputs[name_scope])

        model_output_dict[name_scope] = output_layer
        name_scope_list.append(name_scope)
        model_layers_built[name_scope] = prime_layers

        if len(params['group_layers']) > 1 and 'input' not in params['group_layers']:
            while True:
                name_scope_list = list(set(name_scope_list))
                new_model_output_dict = dict()
                new_name_scope_list = []
                group_names = set()
                for name_branch in sorted(name_scope_list):
                    name_split = name_branch.split('_')
                    if name_split[0] == 'rule':
                        name_split = ['_'.join(name_split[:2])] + name_split[2:]
                    if 'input' in name_split:
                        raise ValueError(
                            "keyword 'input' is not allowed to name a group or name_scope when have more than "
                            "one group_layer")
                    if len(name_split) > 1:
                        new_name_scope = '_'.join(name_split[:-1])
                        new_name_scope_list.append(new_name_scope)
                        group_names.add(new_name_scope)
                if len(new_name_scope_list) > 0:
                    name_scope_list = copy.deepcopy(new_name_scope_list)
                else:
                    break
                for group_name in sorted(group_names):
                    input_list = []
                    for name_branch in sorted(model_output_dict.keys()):
                        if group_name + '_' in name_branch:
                            input_list.append(model_output_dict[name_branch])
                    input_branch = tf.keras.layers.concatenate(input_list, 1, name=f'concat_inp_layer_{group_name}')
                    shape = input_branch.get_shape().as_list()[1:]
                    input_name = 'concat_input_' + group_name
                    model_branch, layers_branch_built = build_layer(input_name, shape,
                                                                    model_layers['output'], group_name,
                                                                    params, train=train)
                    output_branch = model_branch(input_branch)
                    new_model_output_dict[group_name] = output_branch
                    model_layers_built[group_name] = layers_branch_built
                if len(new_model_output_dict) > 0:
                    model_output_dict = new_model_output_dict
        input_list = []
        for group_name in sorted(model_output_dict.keys()):
            input_list.append(model_output_dict[group_name])
        if len(input_list) > 1:
            model_output = tf.keras.layers.concatenate(input_list, 1, name=f'concat_layers_for_output')
        elif len(input_list) == 1:
            model_output = input_list[0]
        else:
            raise RuntimeError('Failed to build model output')

    if is_global:
        with tf.name_scope("clustering") as scope:
            if is_fuzzy:
                shape = tuple(x['clustering'].shape[1:])
                inp_act = tf.keras.Input(shape, name='clustering')
                fuzzy_layer = RBFLayer(params['thres_act'], params['centroids'], params['var_init'])
                model_layers_built['clustering'] = fuzzy_layer
                act_all = tf.keras.layers.concatenate(fuzzy_layer(inp_act), axis=-1, name='activations')
            else:
                inp_act = [tf.keras.Input((1,), name=f'act_{rule}') for rule in sorted(params['rules'])]
                act_all = tf.keras.layers.concatenate(inp_act, axis=-1, name='activations')
            inputs['clustering'] = inp_act
            act_nan_model = act_nan_layer(params['thres_act'])
            act_nan_err = act_nan_model(act_all)
            apply_activations_layer = apply_activations(params['thres_act'])
            model_output = apply_activations_layer([model_output, act_all])
    else:
        act_all = tf.constant(0, dtype=tf.float32, name='activations')
        act_nan_err = tf.constant(0, dtype=tf.float32, name='act_nan_err')
    shape = tuple(model_output.get_shape().as_list()[1:])
    name_scope = 'output_scope'
    layer_output, layers_built = build_layer('input_output_scope', shape, model_layers['output'],
                                             name_scope, params, train=train)
    model_output = layer_output(model_output)
    model_layers_built[name_scope] = layers_built
    name_scope = 'output'
    if not probabilistic:
        with tf.name_scope(name_scope) as scope:
            shape = tuple(model_output.get_shape().as_list()[1:])
            layer_output = layers_functions['dense']('inp_output', shape, {'act_func': None}, params['n_out'],
                                                     'output', train=train)
            model_output = layer_output(model_output)
            model_output = tf.keras.layers.Add()([model_output, tf.expand_dims(act_nan_err, -1)])
        model_layers_built['output'] = dict()
        model_layers_built['output'][layer_output.name] = layer_output
    else:
        with tf.name_scope(name_scope) as scope:
            shape = tuple(model_output.get_shape().as_list()[1:])
            proba_layer_output, layer_output = proba_output(shape, quantiles)
            model_output = proba_layer_output(model_output)
        model_layers_built['output'] = dict()
        for layer in layer_output:
            model_layers_built['output'][layer.name] = layer

    global_model = tf.keras.Model(inputs, model_output)
    return global_model, model_layers_built, act_all

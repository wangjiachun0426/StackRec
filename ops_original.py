import tensorflow as tf
import math
import numpy as np


def get_parameters(train, reader, variable_name, new_variable_name):
    if train:
        print(variable_name, "  --->  ", new_variable_name)
    param = reader.get_tensor(variable_name)
    return param


def nextitnet_residual_block(input_, dilation, layer_id, method,
                                    residual_channels, kernel_size, reader, layer_num, train=True):
    resblock_type = "decoder"
    resblock_name = "nextitnet_residual_block{}_layer_{}".format(resblock_type, layer_id)

    with tf.variable_scope(resblock_name, reuse=tf.AUTO_REUSE):

        if reader:
            dilated_conv = conv1d(input_, residual_channels, method, layer_id, reader, layer_num, resblock_name,
                                  dilation, kernel_size,
                                  name="dilated_conv1",
                                  trainable=train
                                  )
            input_ln = layer_norm(dilated_conv, layer_id, reader, method, layer_num, resblock_name, name="layer_norm1",
                                  trainable=train)
            relu1 = tf.nn.relu(input_ln)
        else:
            dilated_conv = conv1d(input_, residual_channels, method, layer_id, None, layer_num, resblock_name,
                                  dilation, kernel_size,
                                  name="dilated_conv1",
                                  trainable=train
                                  )
            input_ln = layer_norm(dilated_conv, layer_id, None, method, layer_num, resblock_name, name="layer_norm1",
                                  trainable=train)
            relu1 = tf.nn.relu(input_ln)

        if reader:
            dilated_conv = conv1d(relu1, residual_channels, method, layer_id, reader, layer_num, resblock_name,
                                  2 * dilation, kernel_size,
                                  name="dilated_conv2",
                                  trainable=train
                                  )
            input_ln = layer_norm(dilated_conv, layer_id, reader, method, layer_num, resblock_name, name="layer_norm2",
                                  trainable=train)
            relu1 = tf.nn.relu(input_ln)
        else:
            dilated_conv = conv1d(relu1, residual_channels, method, layer_id, None, layer_num, resblock_name,
                                  2 * dilation, kernel_size,
                                  name="dilated_conv2",
                                  trainable=train
                                  )
            input_ln = layer_norm(dilated_conv, layer_id, None, method, layer_num, resblock_name, name="layer_norm2",
                                  trainable=train)
            relu1 = tf.nn.relu(input_ln)

        return input_ + relu1


def conv1d(input_, output_channels, method, layer_id, reader, layer_num, resblock_name,
           dilation=1, kernel_size=1,
           name="dilated_conv", trainable=True):
    with tf.variable_scope(name):

        if method == 'from_scratch' or method == 'stackE':
            weight = tf.get_variable('weight', [1, kernel_size, input_.get_shape()[-1], output_channels],
                                     initializer=tf.truncated_normal_initializer(stddev=0.02, seed=1))
            bias = tf.get_variable('bias', [output_channels],
                                   initializer=tf.constant_initializer(0.0))

        if method == 'stackC':
            if layer_id >= layer_num / 2:
                relative_layer_id = layer_id % int(layer_num / 2)
                variable_name = resblock_name.split("_")
                variable_name[4] = str(relative_layer_id)
                variable_name = "_".join(variable_name) + '/' + name + '/weight'
            else:
                variable_name = resblock_name + '/' + name + '/weight'

            new_variable_name = resblock_name + '/' + name + '/weight'
            initial_value = get_parameters(trainable, reader, variable_name, new_variable_name)

            weight = tf.get_variable('weight', [1, kernel_size, input_.get_shape()[-1], output_channels],
                                     initializer=tf.constant_initializer(initial_value, verify_shape=True))

            if layer_id >= layer_num / 2:
                relative_layer_id = layer_id % int(layer_num / 2)
                variable_name = resblock_name.split("_")
                variable_name[4] = str(relative_layer_id)
                variable_name = "_".join(variable_name) + '/' + name + '/bias'
            else:
                variable_name = resblock_name + '/' + name + '/bias'
            new_variable_name = resblock_name + '/' + name + '/bias'
            initial_value = get_parameters(trainable, reader, variable_name, new_variable_name)
            bias = tf.get_variable('bias', [output_channels],
                                   initializer=tf.constant_initializer(initial_value, verify_shape=True))

        if method == 'StackR':
            if layer_id >= layer_num / 2:
                weight = tf.get_variable('weight', [1, kernel_size, input_.get_shape()[-1], output_channels],
                                         initializer=tf.truncated_normal_initializer(stddev=0.02, seed=1))
            else:
                variable_name = resblock_name + '/' + name + '/weight'

                new_variable_name = resblock_name + '/' + name + '/weight'
                initial_value = get_parameters(trainable, reader, variable_name, new_variable_name)

                weight = tf.get_variable('weight', [1, kernel_size, input_.get_shape()[-1], output_channels],
                                         initializer=tf.constant_initializer(initial_value, verify_shape=True))
            if layer_id >= layer_num / 2:
                bias = tf.get_variable('bias', [output_channels],
                                       initializer=tf.constant_initializer(0.0))
            else:
                variable_name = resblock_name + '/' + name + '/bias'
                new_variable_name = resblock_name + '/' + name + '/bias'
                initial_value = get_parameters(trainable, reader, variable_name, new_variable_name)
                bias = tf.get_variable('bias', [output_channels],
                                       initializer=tf.constant_initializer(initial_value, verify_shape=True))

        if method == 'stackA':
            relative_layer_id = layer_id // 2
            variable_name = resblock_name.split("_")
            variable_name[4] = str(relative_layer_id)
            variable_name = "_".join(variable_name) + '/' + name + '/weight'

            new_variable_name = resblock_name + '/' + name + '/weight'
            initial_value = get_parameters(trainable, reader, variable_name, new_variable_name)

            weight = tf.get_variable('weight', [1, kernel_size, input_.get_shape()[-1], output_channels],
                                     initializer=tf.constant_initializer(initial_value, verify_shape=True))

            relative_layer_id = layer_id // 2
            variable_name = resblock_name.split("_")
            variable_name[4] = str(relative_layer_id)
            variable_name = "_".join(variable_name) + '/' + name + '/bias'

            new_variable_name = resblock_name + '/' + name + '/bias'
            initial_value = get_parameters(trainable, reader, variable_name, new_variable_name)
            bias = tf.get_variable('bias', [output_channels],
                                   initializer=tf.constant_initializer(initial_value, verify_shape=True))

        padding = [[0, 0], [(kernel_size - 1) * dilation, 0], [0, 0]]
        padded = tf.pad(input_, padding)
        input_expanded = tf.expand_dims(padded, dim=1)
        out = tf.nn.atrous_conv2d(input_expanded, weight, rate=dilation, padding='VALID') + bias

        return tf.squeeze(out, [1])


def layer_norm(x, layer_id, reader, method, layer_num, resblock_name, name, epsilon=1e-8, trainable=True):
    with tf.variable_scope(name):
        shape = x.get_shape()

        if method == 'from_scratch':
            beta = tf.get_variable('beta', [int(shape[-1])],
                                   initializer=tf.constant_initializer(0), trainable=trainable)
            gamma = tf.get_variable('gamma', [int(shape[-1])],
                                    initializer=tf.constant_initializer(1), trainable=trainable)

        if method == 'stackC' or method == 'stackA' or method == 'stackE':
            beta = tf.get_variable('beta', [int(shape[-1])],
                                   initializer=tf.constant_initializer(0), trainable=trainable)
            gamma = tf.get_variable('gamma', [int(shape[-1])],
                                    initializer=tf.constant_initializer(1), trainable=trainable)

        if method == 'StackR':
            if layer_id >= layer_num / 2:
                beta = tf.get_variable('beta', [int(shape[-1])],
                                       initializer=tf.constant_initializer(0), trainable=trainable)
            else:
                variable_name = resblock_name + '/' + name + '/beta'
                new_variable_name = resblock_name + '/' + name + '/beta'
                initial_value = get_parameters(trainable, reader, variable_name, new_variable_name)
                beta = tf.get_variable('beta', [int(shape[-1])],
                                       initializer=tf.constant_initializer(initial_value, verify_shape=True),
                                       trainable=trainable)

            if layer_id >= layer_num / 2:
                gamma = tf.get_variable('gamma', [int(shape[-1])],
                                        initializer=tf.constant_initializer(1), trainable=trainable)
            else:
                variable_name = resblock_name + '/' + name + '/gamma'
                new_variable_name = resblock_name + '/' + name + '/gamma'
                initial_value = get_parameters(trainable, reader, variable_name, new_variable_name)
                gamma = tf.get_variable('gamma', [int(shape[-1])],
                                        initializer=tf.constant_initializer(initial_value, verify_shape=True),
                                        trainable=trainable)

        mean, variance = tf.nn.moments(x, axes=[len(shape) - 1], keep_dims=True)
        x = (x - mean) / tf.sqrt(variance + epsilon)
        return gamma * x + beta

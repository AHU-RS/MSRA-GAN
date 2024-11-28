import tensorflow as tf
import numpy as np
from tensorflow.contrib import layers

# Attention Module (Channel and Spatial Attention)
def attention_module(upconv, skipconv):
    # Channel Attention
    avg_pool = tf.reduce_mean(skipconv, axis=[1, 2], keepdims=True)
    max_pool = tf.reduce_max(skipconv, axis=[1, 2], keepdims=True)
    mlp_shared_layer = tf.layers.Dense(units=skipconv.get_shape()[-1] // 8, activation=tf.nn.relu)

    # Apply MLP on average and max pool results
    mlp_avg_pool = mlp_shared_layer(avg_pool)
    mlp_max_pool = mlp_shared_layer(max_pool)

    # Combine the results
    channel_attention = tf.add(mlp_avg_pool, mlp_max_pool)

    # Apply a 1x1 convolution to refine the channel attention map
    channel_attention = tf.layers.conv2d(channel_attention, filters=skipconv.get_shape()[-1], kernel_size=1,
                                         activation=tf.nn.sigmoid)

    # Apply the channel attention to the skip connection
    channel_refined_skipconv = skipconv * channel_attention

    # Spatial Attention
    avg_pool_spatial = tf.reduce_mean(channel_refined_skipconv, axis=-1, keepdims=True)
    max_pool_spatial = tf.reduce_max(channel_refined_skipconv, axis=-1, keepdims=True)
    spatial_attention = tf.concat([avg_pool_spatial, max_pool_spatial], axis=-1)

    # Apply a 7x7 convolution to refine the spatial attention map
    spatial_attention = tf.layers.conv2d(spatial_attention, filters=1, kernel_size=7, padding='same',
                                         activation=tf.nn.sigmoid)

    # Apply the spatial attention to the refined skip connection
    refined_skipconv = channel_refined_skipconv * spatial_attention

    # Fuse the refined skip connection with the upconv using element-wise multiplication and addition
    outputs = tf.multiply(refined_skipconv, tf.nn.sigmoid(upconv), name="fuse_mul")
    outputs = tf.add(upconv, outputs, name="fuse_add")
    return outputs

# Weight initialization with L2 regularization
def weight_variable(shape, name):
    nl = shape[0] * shape[1] * shape[3]
    std = np.sqrt(2 / nl)
    initial = tf.truncated_normal(shape, mean=0, stddev=std)
    weight = tf.Variable(initial_value=initial, name=name)
    tf.add_to_collection("loss", tf.contrib.layers.l2_regularizer(0.001)(weight))
    return weight

# Deconvolution weight initialization
def weight_variable_devonc(shape, stddev=0.001):
    return tf.Variable(tf.truncated_normal(shape, mean=0, stddev=stddev))

# Bias initialization
def bias_variable(shape, name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial_value=initial, name=name)

# 2D Convolution operation
def conv2d(x, w, s=1):
    return tf.nn.conv2d(x, w, strides=[1, s, s, 1], padding='SAME')

# ReLU activation function
def relu(x):
    return tf.nn.relu(x, name='relu')


# Dilation layer (atrous convolution)
def dilation_layer(x, input_channel, output_channel, k_size=3, rate=2, padding='SAME'):
    w = weight_variable([k_size, k_size, input_channel, output_channel], 'weight')
    b = bias_variable([output_channel], 'bias')
    dilation_result = tf.nn.atrous_conv2d(x, w, rate, padding) + b
    return dilation_result


# Regular convolution layer
def conv_layer(x, input_channel, output_channel, k_size=3, stride=1):
    w = weight_variable([k_size, k_size, input_channel, output_channel], 'weight')
    b = bias_variable([output_channel], 'bias')
    conv_result = conv2d(x, w, stride) + b
    return conv_result

# Deconvolution (transpose convolution) operation
def deconv2d(x, stride):
    x_shape = x.get_shape().as_list()
    wshape = [2, 2, x_shape[3] // 2, x_shape[3]]
    x_shape[1] *= 2
    x_shape[2] *= 2
    x_shape[3] //= 2
    W = weight_variable_devonc(wshape, 0.001)
    return tf.nn.conv2d_transpose(x, W, x_shape, strides=[1, stride, stride, 1], padding='VALID')

# Residual block with two convolutional layers
def residual_block(x, channels, is_training, kernel_size=3, stride=1, dropout_rate=0.1):
    identity = x
    w = weight_variable([kernel_size, kernel_size, channels, channels], 'res_w1')
    x = tf.nn.conv2d(x, w, strides=[1, stride, stride, 1], padding='SAME')
    x = tf.nn.relu(tf.layers.batch_normalization(x, training=is_training))

    w = weight_variable([kernel_size, kernel_size, channels, channels], 'res_w2')
    x = tf.nn.conv2d(x, w, strides=[1, stride, stride, 1], padding='SAME')
    x = tf.layers.batch_normalization(x, training=is_training)

    # Skip connection (residual connection)
    x += identity
    return tf.nn.relu(x)

# Generator Network
def Generator(input_1, input_2, input_channel=1, is_bntraining=True):
    # Concatenate and add input tensors
    concat_1_2 = tf.concat([input_1, input_2], axis=3)
    add_1_2 = input_1 + input_2
    concat_1_2 = tf.concat([concat_1_2, add_1_2], axis=3)

    # First block
    conv1_1 = conv_layer(concat_1_2, input_channel, 64, 3, 1)
    conv1_1 = tf.layers.batch_normalization(conv1_1, training=is_bntraining)
    conv1_1 = relu(conv1_1)
    conv1_2 = conv_layer(conv1_1, 64, 64, 3, 1)
    conv1_2 = tf.layers.batch_normalization(conv1_2, training=is_bntraining)
    conv1_2 = relu(conv1_2)

    # Second block
    conv2_1 = tf.nn.avg_pool(conv1_2, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID')
    conv2_2 = conv_layer(conv2_1, 64, 128, 3, 1)
    conv2_2 = tf.layers.batch_normalization(conv2_2, training=is_bntraining)
    conv2_2 = relu(conv2_2)
    conv2_3 = conv_layer(conv2_2, 128, 128, 3, 1)
    conv2_3 = tf.layers.batch_normalization(conv2_3, training=is_bntraining)
    conv2_3 = relu(conv2_3)

    # Third block
    conv3_1 = tf.nn.avg_pool(conv2_3, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID')
    conv3_2 = conv_layer(conv3_1, 128, 256, 3, 1)
    conv3_2 = tf.layers.batch_normalization(conv3_2, training=is_bntraining)
    conv3_2 = relu(conv3_2)
    conv3_3 = conv_layer(conv3_2, 256, 256, 3, 1)
    conv3_3 = tf.layers.batch_normalization(conv3_3, training=is_bntraining)
    conv3_3 = relu(conv3_3)

    # Fourth block
    conv4_1 = tf.nn.avg_pool(conv3_3, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID')
    conv4_2 = conv_layer(conv4_1, 256, 512, 3, 1)
    conv4_2 = tf.layers.batch_normalization(conv4_2, training=is_bntraining)
    conv4_2 = relu(conv4_2)
    conv4_3 = conv_layer(conv4_2, 512, 512, 3, 1)
    conv4_3 = tf.layers.batch_normalization(conv4_3, training=is_bntraining)
    conv4_3 = relu(conv4_3)

    # Fifth block
    conv5_1 = tf.nn.avg_pool(conv4_3, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID')
    conv5_2 = conv_layer(conv5_1, 512, 1024, 3, 1)
    conv5_2 = tf.layers.batch_normalization(conv5_2, training=is_bntraining)
    conv5_2 = relu(conv5_2)
    conv5_3 = conv_layer(conv5_2, 1024, 1024, 3, 1)
    conv5_3 = tf.layers.batch_normalization(conv5_3, training=is_bntraining)
    conv5_3 = relu(conv5_3)

    # Residual block and deconvolution
    conv5_3 = residual_block(conv5_3, 1024, is_bntraining)
    conv54 = deconv2d(conv5_3, 2)
    conv4_5 = attention_module(conv54, conv4_3)
    conv4_6 = conv_layer(conv4_5, 512, 512, 3, 1)
    conv4_6 = tf.layers.batch_normalization(conv4_6, training=is_bntraining)
    conv4_6 = relu(conv4_6)
    conv4_7 = conv_layer(conv4_6, 512, 512, 3, 1)
    conv4_7 = tf.layers.batch_normalization(conv4_7, training=is_bntraining)
    conv4_7 = relu(conv4_7)

    # Deconvolution and attention module
    conv43 = deconv2d(conv4_7, 2)
    conv3_4 = attention_module(conv43, conv3_3)
    conv3_5 = conv_layer(conv3_4, 256, 256, 3, 1)
    conv3_5 = tf.layers.batch_normalization(conv3_5, training=is_bntraining)
    conv3_5 = relu(conv3_5)
    conv3_6 = conv_layer(conv3_5, 256, 256, 3, 1)
    conv3_6 = tf.layers.batch_normalization(conv3_6, training=is_bntraining)
    conv3_6 = relu(conv3_6)

    # Further deconvolution and attention modules
    conv32 = deconv2d(conv3_6, 2)
    conv2_4 = attention_module(conv32, conv2_3)
    conv2_5 = conv_layer(conv2_4, 128, 128, 3, 1)
    conv2_5 = tf.layers.batch_normalization(conv2_5, training=is_bntraining)
    conv2_5 = relu(conv2_5)
    conv2_6 = conv_layer(conv2_5, 128, 128, 3, 1)
    conv2_6 = tf.layers.batch_normalization(conv2_6, training=is_bntraining)
    conv2_6 = relu(conv2_6)

    # Final deconvolution and attention module
    conv21 = deconv2d(conv2_6, 2)
    conv1_3 = attention_module(conv21, conv1_2)
    conv1_4 = conv_layer(conv1_3, 64, 64, 3, 1)
    conv1_4 = tf.layers.batch_normalization(conv1_4, training=is_bntraining)
    conv1_4 = relu(conv1_4)
    conv1_5 = conv_layer(conv1_4, 64, 64, 3, 1)
    conv1_5 = tf.layers.batch_normalization(conv1_5, training=is_bntraining)
    conv1_5 = relu(conv1_5)

    # Final output layer
    conv1_6 = conv_layer(conv1_5, 64, 32, 3, 1)
    conv1_6 = tf.layers.batch_normalization(conv1_6, training=is_bntraining)
    conv1_6 = relu(conv1_6)
    conv1_7 = conv_layer(conv1_6, 32, 1, 3, 1)

    return conv1_7

# Discriminator Network
def Discriminator(input, is_train=False, reuse=False):
    with tf.variable_scope("Discriminator", reuse=tf.AUTO_REUSE):
        # First block: Convolution layers with leaky ReLU activation
        conv1_1 = conv_layer(input, 1, 64, 3, 1)
        conv1_1 = tf.nn.leaky_relu(conv1_1)
        conv1_2 = conv_layer(conv1_1, 64, 64, 3, 2)
        conv1_2 = layers.instance_norm(conv1_2)
        conv1_2 = tf.nn.leaky_relu(conv1_2)

        # Second block
        conv2_2 = conv_layer(conv1_2, 64, 128, 3, 1)
        conv2_2 = layers.instance_norm(conv2_2)
        conv2_2 = tf.nn.leaky_relu(conv2_2)
        conv2_3 = conv_layer(conv2_2, 128, 128, 3, 2)
        conv2_3 = layers.instance_norm(conv2_3)
        conv2_3 = tf.nn.leaky_relu(conv2_3)

        # Third block
        conv3_2 = conv_layer(conv2_3, 128, 256, 3, 1)
        conv3_2 = layers.instance_norm(conv3_2)
        conv3_2 = tf.nn.leaky_relu(conv3_2)
        conv3_3 = conv_layer(conv3_2, 256, 256, 3, 2)
        conv3_3 = layers.instance_norm(conv3_3)
        conv3_3 = tf.nn.leaky_relu(conv3_3)

        # Fourth block
        conv4_2 = conv_layer(conv3_3, 256, 512, 3, 1)
        conv4_2 = layers.instance_norm(conv4_2)
        conv4_2 = tf.nn.leaky_relu(conv4_2)
        conv4_3 = conv_layer(conv4_2, 512, 512, 3, 2)
        conv4_3 = layers.instance_norm(conv4_3)
        conv4_3 = tf.nn.leaky_relu(conv4_3)

        # Final output layer
        logits = layers.linear(conv4_3, 1)
        return logits
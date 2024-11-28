import tensorflow as tf
from utils import read_data
from libs.models.nets_factory import get_network
from keras.applications.vgg16 import VGG16
from keras import backend as K
import model
import time
import os
import argparse
import cv2 as cv
import numpy as np

# Main function for training
def main(args):
    # Define paths for logs and model checkpoints
    logs_path = 'The storage path for  logs files.'
    model_path = 'The storage path for the model.'
    ckpt_path = 'The storage path for checkpoints (HDF5 files).'

    # Hyperparameters
    image_size = 64
    batch_size = 32
    epoch = 60
    epsilon = 1e-8
    learning_rate_value = 1e-4

    # TensorFlow placeholders
    x1 = tf.placeholder(tf.float32, [batch_size, image_size, image_size, 1])
    x2 = tf.placeholder(tf.float32, [batch_size, image_size, image_size, 1])
    y = tf.placeholder(tf.float32, [batch_size, image_size, image_size, 1])
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')

    # Model prediction and losses
    pred = model.Generator(x1, x2, 3, True)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    global_step = tf.get_variable('global_step', [], dtype=tf.int32, initializer=tf.constant_initializer(0), trainable=False)

    # Gradient penalty for WGAN-GP
    def gradient_penalty(real, fake):
        eps = tf.random_uniform([batch_size, 1, 1, 1], minval=0., maxval=1.)
        interpolates = eps * real + (1 - eps) * fake
        gradients = tf.gradients(Discriminator(interpolates, reuse=True), [interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=1))
        return 10 * tf.reduce_mean((slopes - 1.) ** 2)

    # VGG loss function
    def VGGloss(y_true, y_pred):
        vgg_model = VGG16(include_top=False, weights='imagenet')
        pred = K.concatenate([y_pred, y_pred, y_pred])
        true = K.concatenate([y_true, y_true, y_true])
        f_p = vgg_model(pred)
        f_t = vgg_model(true)
        return K.mean(K.square(f_p - f_t))

    # Loss functions
    mse_loss = tf.reduce_mean(tf.sqrt(tf.square(tf.subtract(y, pred)) + epsilon))
    vgg_loss = VGGloss(y, pred)
    G_loss = mse_loss + 5 * (10 ** -7) * vgg_loss
    tf.add_to_collection("G_loss", G_loss)
    G_loss = tf.add_n(tf.get_collection("G_loss"))

    Gan_loss = tf.reduce_mean(Discriminator(y) - Discriminator(pred))
    GP_loss = gradient_penalty(y, pred)
    D_loss = 1 * (10 ** -3) * (Gan_loss + 2 * GP_loss)
    tf.add_to_collection("D_loss", D_loss)
    D_loss = tf.add_n(tf.get_collection("D_loss"))

    # Learning rate decay
    learning_rate_node = tf.train.exponential_decay(learning_rate=learning_rate,
                                                    global_step=global_step,
                                                    decay_steps=batch_size,
                                                    decay_rate=0.94,
                                                    staircase=True)

    # Optimizer
    opt = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.0, beta2=0.9, name='optimizer')
    with tf.control_dependencies(update_ops):
        grads_G = opt.compute_gradients(G_loss)
        grads_D = opt.compute_gradients(D_loss)
        grads_M = opt.compute_gradients(mse_loss)
        train_op_G = opt.apply_gradients(grads_G, global_step=global_step)
        train_op_D = opt.apply_gradients(grads_D, global_step=global_step)
        train_op_M = opt.apply_gradients(grads_M, global_step=global_step)

    # TensorFlow session configuration
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    tf_config.allow_soft_placement = True
    sess = tf.InteractiveSession(config=tf_config)
    sess.run(tf.global_variables_initializer())

    # Variable list for saving
    var_list = tf.trainable_variables()
    if global_step is not None:
        var_list.append(global_step)
    g_list = tf.global_variables()
    bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name or 'moving_variance' in g.name]
    var_list += bn_moving_vars

    # Saver for model checkpoints
    saver = tf.train.Saver(var_list=var_list, max_to_keep=10)
    if not os.path.exists(model_path):
        os.mkdir(model_path)

    # Load data
    input1, input2, label = read_data(ckpt_path)
    summary_writer = tf.summary.FileWriter(logs_path, sess.graph)

    # Training loop
    epoch_learning_rate = learning_rate_value
    for one_ep in range(epoch):
        if one_ep == epoch * 0.3 or one_ep == epoch * 0.5:
            epoch_learning_rate *= 0.5

        train_loss = 0.0
        batch_idxs = len(input1) // batch_size
        for idx in range(batch_idxs):
            batch_input1 = input1[idx * batch_size: (idx + 1) * batch_size]
            batch_input2 = input2[idx * batch_size: (idx + 1) * batch_size]
            batch_label = label[idx * batch_size: (idx + 1) * batch_size]

            # Training the Generator, Discriminator, and MSE
            _, errG = sess.run([train_op_G, G_loss], feed_dict={x1: batch_input1, x2: batch_input2, y: batch_label, learning_rate: epoch_learning_rate})
            _, errD = sess.run([train_op_D, D_loss], feed_dict={x1: batch_input1, x2: batch_input2, y: batch_label, learning_rate: epoch_learning_rate})
            _, errM = sess.run([train_op_M, mse_loss], feed_dict={x1: batch_input1, x2: batch_input2, y: batch_label, learning_rate: epoch_learning_rate})

            if idx % 100 == 0:
                print(f"Epoch: [{one_ep + 1}], step: [{idx + 1}], time: [{time.time() - start_time:.4f}], G_loss: [{errG:.8f}], D_loss: [{errD:.8f}], MSE_loss: [{errM:.8f}]")

            # Save the model every 500 steps
            if idx % 500 == 0:
                train_loss /= 500.0
                train_summary = tf.Summary(value=[tf.Summary.Value(tag='train_loss', simple_value=train_loss)])
                summary_writer.add_summary(train_summary, global_step=one_ep)
                summary_writer.flush()
                save_path = os.path.join(model_path, f'LST_CNN_itr{idx}.ckpt')
                saver.save(sess, save_path)
                print(f'Model saved at {save_path}')

                print(f"GPU Memory Usage after epoch {one_ep}:")
                print(sess.run(tf.contrib.memory_stats.MaxBytesInUse()))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument("--gpu", choices=['0', '1', '2', '3'], default='0', help="GPU ID")
    args = parser.parse_args()
    main(args)
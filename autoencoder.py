# -*- coding: utf-8 -*-

""" Auto Encoder Example.
Using an auto encoder on MNIST handwritten digits.
References:
    Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. "Gradient-based
    learning applied to document recognition." Proceedings of the IEEE,
    86(11):2278-2324, November 1998.
Links:
    [MNIST Dataset] http://yann.lecun.com/exdb/mnist/
"""
from __future__ import division, print_function, absolute_import

import numpy as np
import tensorflow as tf

from helpers import get_all_pickle_files, pickle_to_numpy_array, get_string_to_index_map

starter_learning_rate = 0.5

# Network Parameters
n_input = len(get_string_to_index_map())  # Number of tags that can be output by the actual network
encoding_size = 24
structure = [n_input, encoding_size]

graph = tf.Graph()

with graph.as_default():
    global_step = tf.Variable(0, trainable=False)
    learning_rate = 0.1  # tf.train.exponential_decay(starter_learning_rate, global_step, 5000, 0.7)

    X = tf.placeholder("float", [None, n_input], name="inputs")

    encoder_weights = []
    encoder_biases = []

    decoder_weights = []
    decoder_biases = []

    for i in range(1, len(structure)):
        encoder_weights.append(tf.Variable(tf.random_normal([structure[i - 1], structure[i]])))
        encoder_biases.append(tf.Variable(tf.random_normal([structure[i]])))

    for i in range(len(structure) - 2, -1, -1):
        decoder_weights.append(tf.Variable(tf.random_normal([structure[i + 1], structure[i]])))
        decoder_biases.append(tf.Variable(tf.random_normal([structure[i]])))


    # Building the encoder
    def encoder(x):
        # Encoder Hidden layer with sigmoid activation #1
        encoded = tf.nn.sigmoid(tf.add(tf.matmul(x, encoder_weights[0]), encoder_biases[0]))
        for i in range(1, len(encoder_weights)):
            encoded = tf.nn.sigmoid(tf.add(tf.matmul(encoded, encoder_weights[i]), encoder_biases[i]))

        encoded = tf.round(encoded, name="binary_code")

        return encoded


    # Building the decoder
    def decoder(x):
        decoded = tf.nn.sigmoid(tf.add(tf.matmul(x, decoder_weights[0]), decoder_biases[0]))

        for i in range(1, len(decoder_weights)):
            decoded = tf.nn.sigmoid(tf.add(tf.matmul(decoded, decoder_weights[i]), decoder_biases[i]))
        return decoded


    # Construct model
    encoder_op = encoder(X)
    decoder_op = decoder(encoder_op)

    # Prediction
    y_pred = decoder_op
    # Targets (Labels) are the input data.
    y_true = X

    # Define loss and optimizer, minimize the squared error
    cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
    optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost, global_step=global_step)

    init = tf.initialize_all_variables()
    saver = tf.train.Saver()


def train(training_epochs=100, display_step=1, batch_size=100, create_new_model=False):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # Launch the graph
    with tf.Session(config=config, graph=graph) as sess:
        if not create_new_model:
            try:
                saver.restore(sess, "models/autoencoder.ckpt")
                print("Found previous model at models/autoencoder.ckpt")
            except ValueError:
                print("Model not found. Initializing new")
                # Initializing the variables
                sess.run(init)
        else:
            print("Initializing new model")
            sess.run(init)

        validation_costs = []
        for f in get_all_pickle_files("validate/pickle", combined=True):
            _, test_xs = pickle_to_numpy_array(f)
            if len(test_xs) == 0:
                continue
            c = sess.run(cost, feed_dict={X: test_xs})
            validation_costs.append(c)

        avg_validation_cost = sum(validation_costs) / len(validation_costs)

        last_validation_cost = avg_validation_cost
        # Training cycle
        training_files = get_all_pickle_files("train/pickle", combined=True)
        for epoch in range(training_epochs):
            # Loop over all batches
            training_costs = []
            for f in training_files:
                _, all_xs = pickle_to_numpy_array(f)
                np.random.shuffle(all_xs)
                for i in range(0, len(all_xs), batch_size):
                    batch_xs = all_xs[i:min(i + batch_size, len(all_xs)), :]
                    # Run optimization op (backprop) and cost op (to get loss value)
                    _, c = sess.run([optimizer, cost], feed_dict={X: batch_xs})
                    training_costs.append(c)
            # Display logs per epoch step
            if epoch % display_step == 0:
                avg_training_cost = sum(training_costs) / len(training_costs)
                print("Epoch:", '%04d' % (epoch + 1),
                      "cost=", "{:.9f}".format(avg_training_cost))

                validation_costs = []
                for f in get_all_pickle_files("validate/pickle"):
                    _, test_xs = pickle_to_numpy_array(f)
                    if len(test_xs) == 0:
                        continue
                    c = sess.run(cost, feed_dict={X: test_xs})
                    validation_costs.append(c)

                avg_validation_cost = sum(validation_costs) / len(validation_costs)
                print("Validation set cost:", "{:.9f}".format(avg_validation_cost))
                if avg_validation_cost > last_validation_cost:
                    print("Validation cost worsened. Breaking training.")
                    break
                else:
                    last_validation_cost = avg_validation_cost
                    saver.save(sess, "models/autoencoder.ckpt")

        print("Optimization Finished!")


def get_binary_classification(image_as_numpy_array):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # Launch the graph
    with tf.Session(config=config, graph=graph) as sess:
        try:
            saver.restore(sess, "models/autoencoder.ckpt")
        except ValueError as e:
            print("Autoencoder model not found")
            raise e

        encoded = sess.run(encoder_op, feed_dict={X: [image_as_numpy_array]})

        code_int_array = []

        for float_num in encoded[0]:
            code_int_array.append(int(float_num))

        return tuple(code_int_array)


if __name__ == "__main__":
    train()

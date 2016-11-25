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

from helpers import get_all_pickle_files, pickle_to_numpy_array

starter_learning_rate = 0.5

# Network Parameters
n_input = 6012  # Number of tags that can be output by the actual network
encoding_size = 14
structure = [n_input, encoding_size]

graph = tf.Graph()

with graph.as_default():
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 5000, 0.7)

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


def train(training_epochs=100, display_step=1):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # Launch the graph
    with tf.Session(config=config, graph=graph) as sess:
        try:
            saver.restore(sess, "models/autoencoder.ckpt")
        except ValueError:
            print("Model not found. Initializing new")
            # Initializing the variables
            sess.run(init)

        costs = []
        for f in get_all_pickle_files("validate/pickle", combined=True):
            _, test_xs = pickle_to_numpy_array(f, n_input)
            if len(test_xs) == 0:
                continue
            c = sess.run(cost, feed_dict={X: test_xs})
            costs.append(c)

        avg_cost = sum(costs) / len(costs)

        last_valid_cost = avg_cost
        # Training cycle
        for epoch in range(training_epochs):
            # Loop over all batches
            for f in get_all_pickle_files("train/pickle"):
                _, all_xs = pickle_to_numpy_array(f, n_input)
                np.random.shuffle(all_xs)
                for i in range(0, len(all_xs), 100):
                    batch_xs = all_xs[i:i + 100, :]
                    # Run optimization op (backprop) and cost op (to get loss value)
                    _, c = sess.run([optimizer, cost], feed_dict={X: batch_xs})
            # Display logs per epoch step
            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch + 1),
                      "cost=", "{:.9f}".format(c))

                costs = []
                for f in get_all_pickle_files("validate/pickle"):
                    _, test_xs = pickle_to_numpy_array(f, n_input)
                    if len(test_xs) == 0:
                        continue
                    c = sess.run(cost, feed_dict={X: test_xs})
                    costs.append(c)

                avg_cost = sum(costs) / len(costs)
                print("Validation set cost:", "{:.9f}".format(avg_cost))
                if avg_cost > last_valid_cost:
                    print("Validation cost worsened. Breaking training.")
                    break
                else:
                    last_valid_cost = avg_cost
                    saver.save(sess, "models/autoencoder.ckpt")

        print("Optimization Finished!")


if __name__ == "__main__":
    train()

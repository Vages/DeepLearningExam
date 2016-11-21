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

import tensorflow as tf
import numpy as np
import os
import pickle

# Parameters
learning_rate = 0.01
training_epochs = 1000
batch_size = 256
display_step = 1
examples_to_show = 10

# Network Parameters
n_input = 6012  # MNIST data input (img shape: 28*28)
encoding_size = 14

# tf Graph input (only pictures)
X = tf.placeholder("float", [None, n_input])

structure = [n_input, encoding_size]

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

    encoded = tf.round(encoded)

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
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

# Initializing the variables
init = tf.initialize_all_variables()

def get_all_pickle_files(train_folder):
    for root, _, files in os.walk(train_folder):
        return [os.path.join(root, file) for file in files if
                file not in ["OLD-all_files.pickle", "combined.pickle"]]

def get_string_to_index_dict():
    string_to_index = dict()
    with open("openimages-dataset/clean-dict.csv", mode="r", encoding="utf8") as f:
        for i, line in enumerate(f.readlines()):
            label = line.strip().split(",")[1][1:-1]
            string_to_index[label] = i

    return string_to_index


string_label_map = get_string_to_index_dict()


def pickle_to_numpy_array(filename, string_map):
    with open(filename, mode="rb") as f:
        pickled_dict = pickle.load(f)

    no_examples = len(pickled_dict)

    data = np.zeros([no_examples, n_input])

    for i, key in enumerate(pickled_dict):
        for label, value in pickled_dict[key]:
            try:
                j = string_map[label]
                data[i, j] = value
            except KeyError:
                continue

    return data


config = tf.ConfigProto()
config.gpu_options.allow_growth = True

# Launch the graph
with tf.Session(config=config) as sess:
    saver = tf.train.Saver()

    try:
        saver.restore(sess, "models/autoencoder.ckpt")
    except ValueError:
        print("Model not found. Initializing new")
        sess.run(init)

    costs = []
    for f in get_all_pickle_files("validate/pickle"):
        test_xs = pickle_to_numpy_array(f, string_label_map)
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
            batch_xs = pickle_to_numpy_array(f, string_label_map)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={X: batch_xs})
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch + 1),
                  "cost=", "{:.9f}".format(c))

            costs = []
            for f in get_all_pickle_files("validate/pickle"):
                test_xs = pickle_to_numpy_array(f, string_label_map)
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



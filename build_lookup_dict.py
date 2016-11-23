from collections import defaultdict

import tensorflow as tf
from autoencoder import get_all_pickle_files, pickle_to_numpy_array

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

# Launch the graph
with tf.Session(config=config) as sess:
    saver = tf.train.Saver()

    saver.restore(sess, "models/autoencoder.ckpt")
    code_to_pic_dic = defaultdict(list)

    binary_layer = sess.graph.get_tensor_by_name("binary_code:0")
    X = sess.graph.get_tensor_by_name("inputs:0")
    for f in get_all_pickle_files("train/pickle"):
        (file_paths, labels), values = pickle_to_numpy_array(f)

        encodings = sess.run(binary_layer, feed_dict={X: values})

        for file, label, code in zip(file_paths, labels, encodings):
            tmp_array = []
            for float_num in code:
                tmp_array.append(int(float_num))
            code_to_pic_dic[tuple(tmp_array)].append((file, label))

print("Done")

import pickle

with open("last-lookup-dictionary.pickle", "wb") as f:
    pickle.dump(code_to_pic_dic, f)

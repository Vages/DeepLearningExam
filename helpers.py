import pickle

import numpy as np
import os


def get_all_pickle_files(train_folder, combined=False):
    for root, _, files in os.walk(train_folder):
        if combined:
            return [os.path.join(root, file) for file in files if file in ["combined.pickle"]]
        else:
            return [os.path.join(root, file) for file in files if
                    file not in ["combined.pickle", "OLD-all_files.pickle"]]


def get_string_to_index_map():
    string_to_index = dict()
    with open("openimages_dataset/clean-dict.csv", mode="r", encoding="utf8") as f:
        for i, line in enumerate(f.readlines()):
            label = line.strip().split(",")[1][1:-1]
            string_to_index[label] = i

    return string_to_index


def pickle_to_numpy_array(filename, n_input, string_map=None):
    if string_map is None:
        string_map = get_string_to_index_map()
    with open(filename, mode="rb") as f:
        pickled_dict = pickle.load(f)

    no_examples = len(pickled_dict)

    name_of_picture_folder = os.path.splitext(filename)[0][-9:]
    name_of_train_or_validate_folder = os.path.split(os.path.split(filename)[0])[0]

    image_files = []
    labels = []
    data = np.zeros([no_examples, n_input])

    for i, key in enumerate(pickled_dict):
        path_to_image = os.path.join(name_of_train_or_validate_folder, "pics", name_of_picture_folder, key + ".jpg")
        image_files.append(path_to_image)

        encoding, encoded_labels = make_numpy_array_for_one_example(pickled_dict[key], string_map)

        data[i] = encoding
        labels.append(encoded_labels)

    return (image_files, labels), data


def make_numpy_array_for_one_example(label_value_tuples, string_to_index_map=None):
    if string_to_index_map is None:
        string_to_index_map = get_string_to_index_map()

    no_of_categories = 6012
    encoding_array = np.zeros([no_of_categories])
    labels_that_were_encoded = []
    for label, value in label_value_tuples:
        try:
            j = string_to_index_map[label]
            encoding_array[j] = value
            labels_that_were_encoded.append(label)
        except KeyError:
            continue

    return encoding_array, labels_that_were_encoded

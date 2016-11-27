from copy import deepcopy
import pickle

encoding_to_image_database_location = "last-lookup-dictionary.pickle"


def mutate_binary_tuple(t):
    mutated_copies = []
    for i in range(len(t)):
        c = list(deepcopy(t))

        if c[i] == 1:
            c[i] = 0
        else:
            c[i] = 1

        mutated_copies.append(tuple(c))

    return mutated_copies


def find_matching_images(labeling_tuple):
    with open(encoding_to_image_database_location, "rb") as f:
        encoding_database = pickle.load(f)
    codes_to_check = [labeling_tuple] + mutate_binary_tuple(labeling_tuple)

    results = []
    for c in codes_to_check:
        image_bucket = encoding_database[c]
        for t in image_bucket:
            results.append(t)

    return results

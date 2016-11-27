import glob

import autoencoder
import helpers
import retriever
from labeler import label


def clean_up_matches(matches, original_label_set):
    working_matches = []

    only_labels = set()

    for l, _ in original_label_set:
        only_labels.add(l)

    for m in matches:
        _, match_labels = m
        if len(only_labels.intersection(set(match_labels))) != 0:
            working_matches.append(m)

    return working_matches


def retrieve_matches(filename):
    threshold = 0.5
    labels = []

    labels = label(filename)

    print(labels)

    if len(labels) == 0:
        return [], []

    label_numpy_array = helpers.make_numpy_array_for_one_example(labels)[0]

    max_element = max(label_numpy_array)

    if 0.9 > max_element > 0:
        label_numpy_array = (label_numpy_array / max_element) * 0.9

    encoding = autoencoder.get_binary_classification(label_numpy_array)

    matches = retriever.find_matching_images(encoding)
    clean_matches = clean_up_matches(matches, labels)
    return matches, clean_matches


if __name__ == "__main__":
    lens_matches = []
    lens_clean_matches = []

    for pic in glob.glob("./validate/pics/000002000/*.jpg"):
        m, c = retrieve_matches(pic)
        lens_matches.append(len(m))
        print(pic)
        print(len(m))
        print(len(c))
        lens_clean_matches.append(len(c))

    avg_matches = sum(lens_matches) / len(lens_matches)
    avg_clean_matches = sum(lens_clean_matches) / len(lens_clean_matches)

    matches_deviations = [(avg_matches - elem)**2 for elem in lens_matches]
    clean_matches_deviations = [(avg_clean_matches - elem)**2 for elem in lens_clean_matches]

    matches_standard_deviation = (sum(matches_deviations) / len(matches_deviations)) ** 0.5
    clean_matches_standard_deviation = (sum(clean_matches_deviations) / len(clean_matches_deviations)) ** 0.5

    print("Average matches:", avg_matches)
    print("Matches standard deviation:", matches_standard_deviation)
    print("")
    print("Average cleaned matches", avg_clean_matches)
    print("Clean matches standard deviation", clean_matches_standard_deviation)

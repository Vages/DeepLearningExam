
labels_that_can_actually_be_output = set()

with open("openimages-dataset/data/2016_08/labelmap.txt", encoding="utf8") as f:
    for line in f.readlines():
        line = line.strip()
        labels_that_can_actually_be_output.add(line)


with open("openimages-dataset/dict.csv", mode="r", encoding="utf8") as i:
    with open("openimages-dataset/clean-dict.csv", "w", encoding="utf8") as o:
        for line in i.readlines():
            clean_line = line.strip().split(",")
            node_name = clean_line[0][1:-1]
            if node_name in labels_that_can_actually_be_output:
                o.write(line)

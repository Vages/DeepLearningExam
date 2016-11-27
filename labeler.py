import os
import tensorflow as tf
import numpy as np

from tensorflow.contrib.slim.python.slim.nets import inception
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import variables
from tensorflow.python.training import saver as tf_saver

from openimages_dataset.tools.classify import LoadLabelMaps

slim = tf.contrib.slim


def PreprocessImage(image_path, central_fraction=0.875, image_size=299):
    """Load and preprocess an image.

    Args:
      image_path: path to an image
      central_fraction: do a central crop with the specified
        fraction of image covered.
      image_size: image size to run inference on
    Returns:
      An ops.Tensor that produces the preprocessed image.
    """
    if not os.path.exists(image_path):
        tf.logging.fatal('Input image does not exist %s', image_path)
    img_data = tf.gfile.FastGFile(image_path).read()

    # Decode Jpeg data and convert to float.
    img = tf.cast(tf.image.decode_jpeg(img_data, channels=3), tf.float32)

    img = tf.image.central_crop(img, central_fraction=central_fraction)
    # Make into a 4D tensor by setting a 'batch size' of 1.
    img = tf.expand_dims(img, [0])
    img = tf.image.resize_bilinear(img,
                                   [image_size, image_size],
                                   align_corners=False)

    # Center the image about 128.0 (which is done during training) and normalize.
    img = tf.mul(img, 1.0 / 127.5)
    return tf.sub(img, 1.0)


def label(image_path, checkpoint="openimages_dataset/data/2016_08/model.ckpt", num_classes=6012,
          labelmap_path="openimages_dataset/data/2016_08/labelmap.txt", dict_path="openimages_dataset/dict.csv",
          threshold=0.5, rounding_digits=1):
    if not os.path.exists(checkpoint):
        tf.logging.fatal(
            'Checkpoint %s does not exist. Have you download it? See tools/download_data.sh',
            checkpoint)
    g = tf.Graph()
    with g.as_default():
        input_image = PreprocessImage(image_path)

        with slim.arg_scope(inception.inception_v3_arg_scope()):
            logits, end_points = inception.inception_v3(
                input_image, num_classes=num_classes, is_training=False)

        predictions = end_points['multi_predictions'] = tf.nn.sigmoid(
            logits, name='multi_predictions')
        init_op = control_flow_ops.group(variables.initialize_all_variables(),
                                         variables.initialize_local_variables(),
                                         data_flow_ops.initialize_all_tables())
        saver = tf_saver.Saver()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        saver.restore(sess, checkpoint)

        # Run the evaluation on the image
        predictions_eval = np.squeeze(sess.run(predictions))

    # Print top(n) results
    labelmap, label_dict = LoadLabelMaps(num_classes, labelmap_path, dict_path)

    top_k = predictions_eval.argsort()[:][::-1]
    returned_labels = []
    for idx in top_k:
        mid = labelmap[idx]
        display_name = label_dict.get(mid, 'unknown')
        score = predictions_eval[idx]
        if score < threshold:
            if returned_labels:
                break
            else:
                threshold -= 0.1
                if threshold < 0.1:
                    break
        returned_labels.append((display_name, score))

    return returned_labels


if __name__ == "__main__":
    print(label("./validate/pics/000000000/0ad919872b0963f9.jpg"))

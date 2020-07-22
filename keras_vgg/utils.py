import tensorflow as tf
import numpy as np

def load_dataset(input_path, batch_size, width=112, height=112):

    # Datset of filenames
    dataset = tf.data.Dataset.from_tensor_slices(input_path)

    # Shuffle with biggest buffer size
    dataset = dataset.shuffle(len(input_path))

    # Convert names to TFRecords
    dataset = dataset.interleave(tf.data.TFRecordDataset, cycle_length=16)

    # Parse    
    dataset = dataset.map( parse_function(width=width, height=height), num_parallel_calls=16)
    
    # Batch
    dataset = dataset.batch(batch_size)
    
    dataset.prefetch(buffer_size=batch_size)

    return dataset


def parse_function(width=112, height=112):


    def _parse_function(example_proto):
        features = {'image_raw': tf.io.FixedLenFeature([], tf.string),
                    'label': tf.io.FixedLenFeature([], tf.int64)}
        features = tf.io.parse_single_example(example_proto, features)
        # You can do more image distortion here for training data
        img = tf.image.decode_jpeg(features['image_raw'])

        # Resize
        img = tf.image.resize(img, (height, width))

        img = tf.reshape(img, shape=(112, 112, 3))

        #img = tf.py_func(random_rotate_image, [img], tf.uint8)
        img = tf.cast(img, dtype=tf.float32)
        img = tf.subtract(img, 127.5)
        img = tf.multiply(img,  0.0078125)
        img = tf.image.random_flip_left_right(img)
        label = tf.cast(features['label'], tf.int64)
        return img, label

    return _parse_function








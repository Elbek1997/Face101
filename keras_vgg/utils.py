import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.vgg19 import preprocess_input as vgg19_preprocess


def load_dataset(input_path, batch_size, width=224, height=224):

    # Dataset of filenames
    dataset = tf.data.Dataset.from_tensor_slices(input_path)

    # Convert names to TFRecords
    dataset = dataset.interleave(tf.data.TFRecordDataset, cycle_length=8)

    # Shuffle
    dataset = dataset.shuffle(buffer_size=batch_size*30, reshuffle_each_iteration=True)

    # Parse    
    dataset = dataset.map( parse_function(width=width, height=height), num_parallel_calls=8)
    
    # Batch
    dataset = dataset.batch(batch_size)
    
    dataset.prefetch(buffer_size=batch_size)

    return dataset.repeat()


def parse_function(width=224, height=224):


    def _parse_function(example_proto):
        features = {'image_raw': tf.io.FixedLenFeature([], tf.string),
                    'label': tf.io.FixedLenFeature([], tf.int64)}
        features = tf.io.parse_single_example(example_proto, features)
        # You can do more image distortion here for training data
        img = tf.image.decode_jpeg(features['image_raw'])

        img = tf.reshape(img, shape=(1, height, width, 3))

        # VGG19 preprocessing
        img = vgg19_preprocess(img)[0]
        
        img = tf.cast(img, dtype=tf.float32)
        
        label = tf.cast(features['label'], tf.int64)
        return img, label

    return _parse_function








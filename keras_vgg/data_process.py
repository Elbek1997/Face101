import numpy as np
import tensorflow as tf

from os import listdir, makedirs
from os.path import isfile, join, exists

from argparse import ArgumentParser
from tqdm import tqdm

import cv2

parser = ArgumentParser()

from joblib import Parallel, delayed



parser.add_argument("--width", type=int, default=224, help="Image width")

parser.add_argument("--height", type=int, default=224, help="Image height")

parser.add_argument("--input_path", type=str, default="dataset/VGGFace2/train", help="Path for input image")

parser.add_argument("--output_path", type=str, default="dataset/VGGFace2/tfrecords/train", help="Path for output path")


args = parser.parse_args()


def load_image(path):

    image = cv2.imread(path)

    image = cv2.resize(image, (args.width, args.height))

    return cv2.imencode(".jpg", image)[1].tobytes()



def write_tfrecord(folder, image_paths, name):

    label = int(folder[1:])
    # label = int(folder)

    # Create dir if not exists
    if not exists(join(args.output_path, folder)):
        makedirs(join(args.output_path, folder))

    # Write tfRecord
    with tf.io.TFRecordWriter(join(join(args.output_path, folder), name)) as writer:

        for path in tqdm(image_paths, desc="Processing %s"%(folder)):

            img = load_image(join(join(args.input_path, folder), path))

            example = tf.train.Example(features=tf.train.Features(feature={
                'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img])),
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
                }))
            writer.write(example.SerializeToString()) 


# Create output path

if not exists(args.output_path):
    makedirs(args.output_path)

# List folders
folders = listdir(args.input_path)

# for i in tqdm(range(len(folders)), desc="Creating TFrecords"):

#     folder = folders[i]

#     images = listdir(join(args.input_path, folder))
    
#     write_tfrecord(folder, images, "t.tfrecord")


Parallel(n_jobs=8)( delayed(write_tfrecord)(folders[i], listdir(join(args.input_path, folders[i])), "t.tfrecord") for i in range(len(folders)) )

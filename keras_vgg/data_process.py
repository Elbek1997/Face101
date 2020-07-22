import numpy as np
import tensorflow as tf

from os import listdir, makedirs
from os.path import isfile, join, exists

from argparse import ArgumentParser
from tqdm import tqdm

import cv2

parser = ArgumentParser()

parser.add_argument("--slice_size", type=int, default=16, help="Batch size for training")

parser.add_argument("--input_path", type=str, default="dataset/train", help="Path for input image")

parser.add_argument("--output_path", type=str, default="dataset/tfrecords/train", help="Path for output path")


args = parser.parse_args()


def load_image(path):

    binary = None

    with open(path, "rb") as f:

        binary = f.read()
    
    return binary


def write_tfrecord(folder, image_paths, name):

    label = int(folder[1:])

    images = [ load_image(join(join(args.input_path, folder), path)) for path in image_paths ]

    # Create dir if not exists
    if not exists(join(args.output_path, folder)):
        makedirs(join(args.output_path, folder))

    # Write tfRecord
    with tf.io.TFRecordWriter(join(join(args.output_path, folder), name)) as writer:

        for img in images:


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

for i in tqdm(range(len(folders)), desc="Creating TFrecords"):

    folder = folders[i]

    images = listdir(join(args.input_path, folder))
    
    # Select slice_size*k (k is int) samples
    images = images[: len(images)//args.slice_size * args.slice_size]

    for i in range(len(images)//args.slice_size):

        write_tfrecord(folder, images[ i*args.slice_size: (i+1)*args.slice_size ], "%02d.tfrecord"%(i+1))

    


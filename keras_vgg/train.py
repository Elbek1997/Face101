from utils import load_dataset
from model import VGG16

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard


import tensorflow as tf
import numpy as np

from tensorflow_addons.losses import triplet_semihard_loss

from argparse import ArgumentParser

import os
import random


parser = ArgumentParser()

parser.add_argument("--width", type=int, default=112, help="Width of input image")

parser.add_argument("--height", type=int, default=112, help="Height of input image")

parser.add_argument("--embeddings", type=int, default=512, help="Embeddings size")


parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")

parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")

parser.add_argument("--epochs", type=int, default=10, help="Epochs to train")




parser.add_argument("--log_path", type=str, default="log/vgg16_112x112_512", help="Path for saving tensorboard log")

parser.add_argument("--weights_path", type=str, default="weights/vgg16_112x112_512", help="Path for saving weights")

parser.add_argument("--train_dataset", type=str, default="dataset/tfrecords/train", help="Path for training tfrecords")



args = parser.parse_args()


# Create dirs if not existant

if not os.path.exists(args.log_path):
    os.mkdir(args.log_path)

if not os.path.exists(args.weights_path):
    os.mkdir(args.weights_path)

# Load TFRecords
print("[INFO] Loading %s"%(args.train_dataset))

# Read list of tfrecord files
list_of_files = []

for folder in os.listdir(args.train_dataset):
    list_of_files.extend( [ os.path.join( os.path.join(args.train_dataset, folder), file)  for file in os.listdir(os.path.join(args.train_dataset, folder)) ] )


steps_per_epoch = len(list_of_files) * 16 //128

random.shuffle(list_of_files)

train_dataset = load_dataset(list_of_files, batch_size=args.batch_size)


# Load Model
model = VGG16(input_shape=(args.height, args.width, 3))


# Compile model
model.compile(optimizer=Adam(args.learning_rate),
    loss=triplet_semihard_loss,
    metrics=['accuracy'])


# ModelCheckpoint callback
model_checkpoint_callback = ModelCheckpoint(filepath=args.weights_path+"/weight_%0.2d.h5", save_freq='epoch')

# Tensorboard Callback
tensorboard_callback = TensorBoard(log_dir=args.log_path, update_freq='epoch', write_graph=False) 


model.fit(x=train_dataset, epochs=args.epochs, steps_per_epoch=steps_per_epoch, callbacks=[model_checkpoint_callback, tensorboard_callback])








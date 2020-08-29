from utils import load_dataset
from model import VGG19, cosine_loss

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, LearningRateScheduler


import tensorflow as tf
import numpy as np

from tensorflow_addons.losses import triplet_semihard_loss

from argparse import ArgumentParser

import os
import random


parser = ArgumentParser()

parser.add_argument("--width", type=int, default=224, help="Width of input image")

parser.add_argument("--height", type=int, default=224, help="Height of input image")

parser.add_argument("--embeddings", type=int, default=512, help="Embeddings size")


parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")

parser.add_argument("--learning_rate", type=float, default=0.0001, help="Learning rate")

parser.add_argument("--epochs", type=int, default=20, help="Epochs to train")

parser.add_argument("--initial_epoch", type=int, default=0, help="Start epoch")



parser.add_argument("--log_path", type=str, default="log/vgg19_face2", help="Path for saving tensorboard log")

parser.add_argument("--weights_path", type=str, default="weights/vgg19_face2", help="Path for saving weights")

parser.add_argument("--train_dataset", type=str, default="dataset/VGGFace2/tfrecords/train", help="Path for training tfrecords")

parser.add_argument("--validation_split", type=float, default=0.1, help="Validation split")

parser.add_argument("--test_dataset", type=str, default="dataset/VGGFace2/tfrecords/test", help="Path for test tfrecords")



args = parser.parse_args()


# Create dirs if not existant

if not os.path.exists(args.log_path):
    os.mkdir(args.log_path)

if not os.path.exists(args.weights_path):
    os.mkdir(args.weights_path)

# Load TFRecords
print("[INFO] Loading %s"%(args.train_dataset))

#region Read list of tfrecord files [Train dataset]
train_files = []

for folder in os.listdir(args.train_dataset):
    train_files.extend( [ os.path.join( os.path.join(args.train_dataset, folder), file)  for file in os.listdir(os.path.join(args.train_dataset, folder)) ] )

train_size = np.sum([ int(os.path.splitext(os.path.basename(file))[0]) for file in train_files])
print("[INFO] Train_size:%d"%train_size)
train_dataset = load_dataset(train_files, batch_size=args.batch_size)

#endregion

# Load TFRecords
print("[INFO] Loading %s"%(args.test_dataset))

# #region Read list of tfrecord files [Test dataset]
# test_files = []

# for folder in os.listdir(args.test_dataset):
#     test_files.extend( [ os.path.join( os.path.join(args.test_dataset, folder), file)  for file in os.listdir(os.path.join(args.test_dataset, folder)) ] )

# test_size = np.sum([ int(os.path.splitext(os.path.basename(file))[0]) for file in test_files])
# print("[INFO] Test_size:%d"%test_size)
# test_dataset = load_dataset(test_files, batch_size=args.batch_size)

# #endregion

# Load Model
model = VGG19(input_shape=(args.height, args.width, 3))


# Compile model
model.compile(optimizer=Adam(args.learning_rate),
    loss=cosine_loss)


# ModelCheckpoint callback
model_checkpoint_callback = ModelCheckpoint(filepath=args.weights_path+"/weight_{epoch:02d}.h5", save_freq='epoch', save_weights_only=True)

# Tensorboard Callback
tensorboard_callback = TensorBoard(log_dir=args.log_path, update_freq='epoch', write_graph=False) 

# ReduceLROnPlateau Callback
learning_rate_scheduler = LearningRateScheduler(lambda epoch, lr: args.learning_rate * 0.5**(epoch//5) )

if args.initial_epoch > 0:
	model.load_weights("%s/weight_%02d.h5"%(args.weights_path, args.initial_epoch))

model.fit(x=train_dataset, 
    initial_epoch=args.initial_epoch, epochs=args.epochs, 
    steps_per_epoch=train_size//args.batch_size, callbacks=[model_checkpoint_callback, tensorboard_callback, learning_rate_scheduler]
    # , validation_data=test_dataset, validation_steps=test_size//args.batch_size
    )








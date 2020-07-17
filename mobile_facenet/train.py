from model import *
from utils import *
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard

from argparse import ArgumentParser


parser = ArgumentParser(description="Process Triplet loss with MNIST dataset")

parser.add_argument("--batch_size", type=int, default=256, help="Batch size for model")
parser.add_argument("--epochs", type=int, default=100, help="Epochs")
parser.add_argument("--embedding_size", type=int, default=256, help="Embedding size")

parser.add_argument("--height", type=int, default=112, help="Image height")
parser.add_argument("--width", type=int, default=112, help="Image width")
parser.add_argument("--channels", type=int, default=3, help="Image channels")

parser.add_argument("--train_list", type=str, default="dataset/train_list.txt", help="File path to train list")
parser.add_argument("--train_path", type=str, default="dataset/train", help="Train images path")

parser.add_argument("--log_path", type=str, default="log/v1")
parser.add_argument("--weight_path", type=str, default="weights/v1")


args = parser.parse_args()


# Create network
model = network( image_shape=(args.height, args.width, args.channels), embedding_size=args.embedding_size)

optimizer = Adam(lr=0.0001)

model.compile(loss=triplet_loss, optimizer=optimizer)


keras.utils.plot_model(model, "models/v1.png", show_shapes=True)




# Load Data

with open(args.train_list, "r") as f:
    image_paths = f.read().splitlines()

labels = [ int(path.split("/")[0][1:]) for path in image_paths]

# Create Data Generator
generator = ImageGenerator(master_path=args.train_path, image_paths=image_paths, 
    image_shape=(args.height, args.width, args.channels), embeddings=args.embedding_size,
    labels=labels, batch_size=args.batch_size, shuffle=True)
 

# ModelCheckPoint
checkpoint = ModelCheckpoint(filepath=args.weight_path+"/weights_{epoch:02d}.h5", save_weights_only=True,
            period=5)

# Tensorboard logger
tensorboard = TensorBoard(log_dir=args.log_path)

# Train

model.fit_generator(generator, epochs=args.epochs, verbose=1, callbacks=[checkpoint, tensorboard],
    shuffle=True)













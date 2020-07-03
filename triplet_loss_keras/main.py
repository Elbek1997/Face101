from model import *
from tensorflow.keras.datasets import mnist

from argparse import ArgumentParser


parser = ArgumentParser(description="Process Triplet loss with MNIST dataset")

parser.add_argument("--batch_size", type=int, default=256, help="Batch size for model")
parser.add_argument("--epochs", type=int, default=50, help="Epochs")
parser.add_argument("--embedding_size", type=int, default=64, help="Embedding size")

args = parser.parse_args()


# Load mnist data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype("float32")/255.
x_test = x_test.astype("float32")/255.

image_shape = (28, 28, 1)

x_val = x_test[:2000, :, :]
y_val = y_test[:2000]


# Create network
model = network(image_shape, embedding_size=args.embedding_size)
optimizer = Adam(lr=0.0001)

model.compile(loss=triplet_loss, optimizer=optimizer)

filepath = "weights/weight_ep{epoch:02d}_BS%d.hdf5" % args.batch_size
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, period=25)
callbacks_list = [checkpoint]

# Uses 'dummy' embeddings + dummy gt labels; removed as soon as they enter the loss function...
dummy_gt_train = np.zeros((len(x_train), args.embedding_size + 1))
dummy_gt_val = np.zeros((len(x_val), args.embedding_size + 1))

x_train = np.reshape(x_train, (len(x_train), x_train.shape[1], x_train.shape[1], 1))
x_val = np.reshape(x_val, (len(x_val), x_train.shape[1], x_train.shape[1], 1))
H = model.fit(x=[x_train,y_train],
            y=dummy_gt_train,
            batch_size=args.batch_size,
            epochs=args.epochs,
            validation_data=([x_val, y_val], dummy_gt_val),
            callbacks=callbacks_list,
            shuffle=True)







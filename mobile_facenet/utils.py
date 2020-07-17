from tensorflow import keras
import cv2
import numpy as np
from os import path


class ImageGenerator(keras.utils.Sequence):

    def __init__(self, master_path, image_paths, image_shape, embeddings, labels, batch_size, shuffle=True):

        self.master_path = master_path
        self.image_paths = np.asarray(image_paths)
        self.image_shape = image_shape

        self.embeddings = embeddings
        self.labels = np.asarray(labels, dtype="float32")
        
        self.length = len(image_paths)
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Indices
        self.indices = np.arange(self.length)

        # Caches
        self.X = np.zeros(shape=(self.batch_size,)+image_shape, dtype="float32")
        self.Y = np.zeros(shape=(self.batch_size, self.embeddings+1), dtype="float32")

        self.on_epoch_end()


    
    def __len__(self):
        return np.ceil(self.length/self.batch_size).astype("int64")


    def on_epoch_end(self):
        
        if self.shuffle:
            np.random.shuffle(self.indices)


    def __getitem__(self, index):

        # Get indices of batch size
        indices = self.indices[index * self.batch_size: (index+1) * self.batch_size]
        length = len(indices)

        # Copy output batch
        labels = self.labels[indices]
        self.Y[:,0] = labels

        #region Copy input batch
        for i in range(length):
            # Create path
            file = str(path.join(self.master_path, self.image_paths[indices[i]]))

            # Load image
            self.X[i] = cv2.resize(cv2.imread(
                file), self.image_shape[:2], interpolation=cv2.INTER_AREA).astype("float32")/255.0
        #endregion

        return (self.X[:length], labels), self.Y[:length]



        

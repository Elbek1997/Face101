import numpy as np
import cv2
import tensorflow as tf
from scipy.spatial import distance

from sklearn.metrics import pairwise_distances as sk_pairwise_distances

# For triplet loss
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import dtypes

def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the 
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it 
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name="prefix")
    return graph

def pairwise_distance(embedding_1, embedding_2):
    # return distance.cosine(embedding_1, embedding_2)
    return sk_pairwise_distances([embedding_1, embedding_2])[0][1]


class Face_Identification:

    def __init__(self, graph_filename="models/embeddings/trained_model.pb"):
        
        # Load graph
        self.net = cv2.dnn.readNetFromTensorflow(graph_filename)

    def generate_embeddings(self, images):

        # Resize and BGR to RGB
        images_input = np.asarray(
            [cv2.cvtColor(cv2.resize(image, (112, 112), 1.0), cv2.COLOR_BGR2RGB) for image in images])

        # Pre process
        images_input = (images_input.astype("float32") - 127.5)/128.0

        # Transpose
        images_input = images_input.transpose((0, 3, 1, 2))

        self.net.setInput(images_input, 'img_inputs')

        out = self.net.forward()

        return out


    def generate_embedding(self, image):

        # Resize image 112x112
        image = cv2.resize(image, (112, 112), 1.0)

        # Swap BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Pre process
        image = (image.astype("float32") - 127.5)/128.0

        # Set 1 batch size
        image = np.reshape(image, (1, 112, 112, 3))

        # Transpose
        image = image.transpose((0, 3, 1, 2))

        self.net.setInput(image, 'img_inputs')

        out = self.net.forward()

        return out[0]



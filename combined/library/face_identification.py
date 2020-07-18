import numpy as np
import cv2
import tensorflow as tf

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

def pairwise_distance(embeddings_1, embedding_2):
    return np.sqrt(np.average(np.square(embedding_2- embeddings_1)))


class Face_Identification:

    def __init__(self, graph_filename="models/embeddings/trained_model.pb", 
        image_input_name="prefix/img_inputs:0", embeddings_name="prefix/embeddings:0", gpu=True):
        
        # Load graph
        self.graph = load_graph(graph_filename)

        #region Session
        if gpu:
            sess = tf.Session(graph=self.graph)
        
        else:
            config = tf.ConfigProto(device_count={'GPU': 0})
            sess = tf.Session(graph=self.graph, config=config)

        self.session = sess
        #endregion
        
        # Input/output tensor
        self.image_input = self.graph.get_tensor_by_name(image_input_name)
        self.embeddings = self.graph.get_tensor_by_name(embeddings_name)


    def generate_embeddings(self, image):

        # Resize image 112x112
        image = cv2.resize(image, (112, 112), 1.0)

        # BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype("float32")

        # Subtract 127.5 and divide by 128
        image = (image - 127.5) / 128.0

        # Resize with 1 batch size
        image = np.reshape(image, (1, 112, 112, 3))

        # Embeddings
        embeddings = self.session.run(self.embeddings, feed_dict={self.image_input: image})[0]

        return embeddings



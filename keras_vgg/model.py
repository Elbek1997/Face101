


from tensorflow.keras.layers import Flatten, Dense, Input, GlobalAveragePooling2D, \
    GlobalMaxPooling2D, Activation, Conv2D, MaxPooling2D, BatchNormalization, \
    AveragePooling2D, Reshape, Permute, multiply, Lambda

from tensorflow.keras.models import Model

from tensorflow.keras.applications import VGG19 as VGG19_backbone

import tensorflow as tf


def VGG19(input_shape=(224, 224, 3), pooling="avg", normalize=True):
    
    # Backbone VGG19
    backbone = VGG19_backbone(include_top=False, input_shape=input_shape)

    # Backbone output
    x = backbone.output

    # Pooling [Max or Average]
    if pooling=="avg":
        x = GlobalAveragePooling2D()(x)
    elif pooling=="max":
        x = GlobalMaxPooling2D()(x)
    

    # Normalize afterwards
    if normalize:
        x = Lambda(lambda embedding: tf.math.l2_normalize(embedding))(x)
    
    model = Model(backbone.inputs, x, name="VGG19_face_embedding")

    return model
    


def cosine_loss(labels, embeddings):

    # Create mask -1 for indexes with same label, 1 for different
    mask = tf.cast( tf.not_equal( tf.map_fn(lambda x: labels-x, labels), 0), dtype=tf.float32 ) * 2.0 - 1.0

    # Normalize embeddings
    embeddings = tf.math.l2_normalize(embeddings, axis=1)

    # Calculate distances 1 - cosine similarity
    distances = (1.0 - tf.matmul(embeddings, tf.transpose(embeddings)))
    
    # Apply mask
    distances = distances * mask

    # Sum over 2.0 * batch_size
    loss = tf.math.reduce_sum(distances) / (2.0 * tf.cast(tf.shape(labels)[0], dtype=tf.float32) )  

    return loss
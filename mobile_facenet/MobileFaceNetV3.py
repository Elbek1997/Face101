'''Importing libraries & configurations''' 
import tensorflow.keras.backend as K 
from tensorflow.keras.layers import Input, Conv2D, DepthwiseConv2D, BatchNormalization, Activation, GlobalAveragePooling2D, Dense, Reshape, Multiply, Add 
from tensorflow.keras.models import Model 

ALPHA = 1.0 
NUM_LABELS = 1000 

'''Building block funcitons''' 
def activation(inputs, act_choice): 
    
    def relu6(inputs): 
        return K.relu(inputs, max_value = 6.0) 
    
    def hard_swish(inputs): 
        return inputs * K.relu(inputs + 3.0, max_value = 6.0) / 6.0 
    
    if act_choice == 'HS': 
        O = Activation(hard_swish)(inputs)
        
    elif act_choice == 'RE': 
        O = Activation(relu6)(inputs)
    
    return O 

def conv_block(inputs, filters, kernel_size, strides, act_choice): 
    
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1 
    
    Z = Conv2D(filters, kernel_size, strides = strides, padding = 'same', use_bias = False)(inputs) 
    Z = BatchNormalization(axis = channel_axis)(Z) 
    A = activation(Z, act_choice) 
    
    return A 

# Squeeze-And-Excite 
def squeeze_block(inputs): 
    
    input_channels = int(inputs.shape[-1]) 
    
    M = GlobalAveragePooling2D()(inputs) 
    M = Dense(input_channels, activation = 'relu')(M) 
    M = Dense(input_channels, activation = 'hard_sigmoid')(M) 
    M = Reshape((1, 1, input_channels))(M) 
    M = Multiply()([inputs, M]) 
    
    return M 

def bottleneck(inputs, filters, kernel_size, e, s, squeeze, act_choice): 
    
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1 
    input_shape = K.int_shape(inputs) 
    
    tchannel = int(e) 
    cchannel = int(ALPHA * filters) 
    
    r = s == 1 and input_shape[3] == filters # ??? 
    
    A1 = conv_block(inputs, tchannel, 1, 1, act_choice) 
    
    Z2 = DepthwiseConv2D(kernel_size, strides = s, depth_multiplier = 1, padding = 'same')(A1) 
    Z2 = BatchNormalization(axis = channel_axis)(Z2) 
    A2 = activation(Z2, act_choice) 
    
    if squeeze: 
        A2 = squeeze_block(A2) 
    
    Z3 = Conv2D(cchannel, 1, strides = 1, padding = 'same')(A2) 
    O = BatchNormalization(axis = channel_axis)(Z3) 
    
    if r: 
        O = Add()([O, inputs]) 
        
    return O 

'''Building the model''' 
def MobileNetV3_small(image_shape, embedding_size): 
    
    X = Input(shape=image_shape) 
    
    M = conv_block(X, 64, 3, 2, 'HS') 
    
    M = bottleneck(M, 16, 3, e = 16, s = 2, squeeze = True, act_choice = 'RE') 
    
    M = bottleneck(M, 24, 3, e = 72, s = 2, squeeze = False, act_choice = 'RE') 
    
    M = bottleneck(M, 24, 3, e = 88, s = 1, squeeze = False, act_choice = 'RE') 
    
    M = bottleneck(M, 40, 5, e = 96, s = 1, squeeze = True, act_choice = 'HS') 
    
    M = bottleneck(M, 40, 5, e = 240, s = 1, squeeze = True, act_choice = 'HS') 
    
    M = bottleneck(M, 40, 5, e = 240, s = 1, squeeze = True, act_choice = 'HS') 
    
    M = bottleneck(M, 48, 5, e = 120, s = 1, squeeze = True, act_choice = 'HS') 
    
    M = bottleneck(M, 48, 5, e = 144, s = 1, squeeze = True, act_choice = 'HS') 
    
    M = bottleneck(M, 96, 5, e = 288, s = 2, squeeze = True, act_choice = 'HS') 
    
    M = bottleneck(M, 96, 5, e = 576, s = 1, squeeze = True, act_choice = 'HS') 
    
    M = bottleneck(M, 96, 5, e = 576, s = 1, squeeze = True, act_choice = 'HS') 
    
    M = conv_block(M, 576, 1, 1, 'HS') 
    
    M = GlobalAveragePooling2D()(M) 
    
    M = Reshape((1, 1, 576))(M) 
    
    M = Conv2D(1280, 1, strides = 1, padding = 'same')(M)
    M = activation(M, 'HS') 
    
    
    M = Conv2D(embedding_size, 1, strides = 1, padding = 'same', activation = 'softmax')(M) 
    Y = Reshape((embedding_size, ))(M) 
        
    model = Model(X, Y, name="MobileFaceNetV3") 
    
    return model 

# model = MobileNetV3_small(True) 
# model.summary() 
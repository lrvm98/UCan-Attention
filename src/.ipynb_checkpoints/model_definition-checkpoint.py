from att_module import attach_attention_module
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, GaussianNoise
from keras.layers import Add, BatchNormalization, Activation, Embedding, MaxPooling2D, AveragePooling2D, ReLU
from keras.layers import Concatenate, Lambda, Permute, SeparableConv2D
from keras.layers.advanced_activations import LeakyReLU, ELU, ThresholdedReLU, Softmax
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv1D, Deconv2D, Conv2DTranspose
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras import losses
from keras.models import load_model

def dilated_cell_module(units, x, activ='elu', conv_module_type=1):

    if conv_module_type ==1:
        # as in steffens - lars 2018
        dc0 = Conv2D(units//4, 1, strides=1, dilation_rate = 1, activation=activ, padding='same')(x)
        dc1 = Conv2D(units//4, 3, strides=1, dilation_rate = 1, activation=activ, padding='same')(x)
        dc2 = Conv2D(units//4, 3, strides=1, dilation_rate = 2, activation=activ, padding='same')(x)
        dc4 = Conv2D(units//4, 3, strides=1, dilation_rate = 4, activation=activ, padding='same')(x)
        m1 = Concatenate(axis=-1, )([dc0, dc1, dc2, dc4])
        return m1

    elif conv_module_type==2:
        dc1 = Conv2D(units//4, 3, strides=1, dilation_rate = 1, activation=activ, padding='same')(x)
        dc2 = Conv2D(units//4, 3, strides=1, dilation_rate = 2, activation=activ, padding='same')(x)
        dc4 = Conv2D(units//4, 3, strides=1, dilation_rate = 4, activation=activ, padding='same')(x)
        dc8 = Conv2D(units//4, 3, strides=1, dilation_rate = 8, activation=activ, border_mode='same')(x)
        m1 = Concatenate(axis=-1, )([dc8, dc1, dc2, dc4])
        dc0 = Conv2D(units, 1, strides=1, dilation_rate = 1, activation=activ, padding='same')(m1)
        return dc0
    elif conv_module_type==3:
        dc1 = Conv2D(units//4, 3, strides=1, dilation_rate = 1, activation=activ, padding='same')(x)
        dc2 = Conv2D(units//4, 3, strides=1, dilation_rate = 2, activation=activ, padding='same')(x)
        dc4 = Conv2D(units//4, 3, strides=1, dilation_rate = 4, activation=activ, padding='same')(x)
        dc8 = Conv2D(units//4, 3, strides=1, dilation_rate = 8, activation=activ, border_mode='same')(x)
        m1 = Concatenate(axis=-1, )([dc8, dc1, dc2, dc4])
        m1 = attach_attention_module(m1,'cbam_block')
        dc0 = Conv2D(units, 1, strides=1, dilation_rate = 1, activation=activ, padding='same')(m1)
        return dc0
    elif conv_module_type==4:
        dc1 = Conv2D(units//4, 3, strides=1, dilation_rate = 1, activation=activ, padding='same')(x)
        dc2 = Conv2D(units//4, 3, strides=1, dilation_rate = 2, activation=activ, padding='same')(x)
        dc4 = Conv2D(units//4, 3, strides=1, dilation_rate = 4, activation=activ, padding='same')(x)
        dc8 = Conv2D(units//4, 3, strides=1, dilation_rate = 8, activation=activ, border_mode='same')(x)
        m1 = Concatenate(axis=-1, )([dc8, dc1, dc2, dc4])
        dc0 = Conv2D(units, 1, strides=1, dilation_rate = 1, activation=activ, padding='same')(m1)
        dc0 = attach_attention_module(dc0,'cbam_block')
        dc0 = Conv2D(units, 1, strides=1, dilation_rate = 1, activation=activ, padding='same')(dc0)
        return dc0
    
def restoration_net(img_rows=None, img_cols=None, activ='elu', conv_module_type=3, img_channels = 4):

    first_input = Input(shape=( img_rows, img_cols, img_channels))

    dcm1 = dilated_cell_module(32, first_input, activ, conv_module_type)
    l1 = Conv2D(32, 3, strides=2, dilation_rate = 1, activation=activ, padding='same')(dcm1) #128
    l1 = dilated_cell_module(64, l1, activ, conv_module_type)
    l2 = Conv2D(32, 3, strides=2, dilation_rate = 1, activation=activ, padding='same')(l1) # 64
    l2 = dilated_cell_module(128, l2, activ, conv_module_type)
    l3 = Conv2D(32, 3, strides=2, dilation_rate = 1, activation=activ, padding='same')(l2) #32
    l3 = dilated_cell_module(256,l3, activ, conv_module_type)

    l3 = upsampling_cell_module(32, l3, 2) #64
    l4 = Concatenate(axis=-1, )([l3, l2])
    l4 = dilated_cell_module(32, l4, activ, conv_module_type)
    l5 = upsampling_cell_module(32, l4, 2) #128
    l5 = dilated_cell_module(32,l5, activ, conv_module_type)
    l6 = Concatenate(axis=-1, )([l1, l5]) #128
    l6 = upsampling_cell_module(32, l6, 2)
    l6 = dilated_cell_module(32,l6, activ, conv_module_type)

    r1 = Conv2D(32, 3, strides=1, dilation_rate = 1, activation=activ, padding='same')(dcm1)
    r2 = Conv2D(32, 3, strides=1, dilation_rate = 1, activation=activ, padding='same')(r1)

    m1 = Concatenate(axis=-1, )([l6, r2])
    m1 = BatchNormalization()(m1)
    c1 = Conv2D(32, 3, strides=1, dilation_rate = 1, activation=activ, padding='same')(m1)
    c2 = Conv2D(32, 1, strides=1, dilation_rate = 1, activation=activ, padding='same')(c1)

    o1 = Conv2D(3, 1, strides=1, activation='relu', name='out', padding='same')(c2)
    model = Model(inputs=first_input, outputs=o1)

    return model

def attention_net(img_rows=None, img_cols=None, activ='elu', conv_module_type=3, img_channels = 4):
    
    first_input = Input(shape=( img_rows, img_cols, img_channels))
    
    l1 = Conv2D(24, 3, strides=1, dilation_rate = 1, activation=activ, padding='same')(first_input)
    l1 = BatchNormalization()(l1)
    
    l2 = Conv2D(24, 3, strides=1, dilation_rate = 2, activation=activ, padding='same')(l1)
    l2 = BatchNormalization()(l2)
    
    l3 = Conv2D(24, 3, strides=1, dilation_rate = 4, activation=activ, padding='same')(l2)
    l3 = BatchNormalization()(l3)
    
    l4 = Conv2D(24, 3, strides=1, dilation_rate = 8, activation=activ, padding='same')(l3)
    l4 = BatchNormalization()(l4)
    
    l5 = Conv2D(24, 3, strides=1, dilation_rate = 16, activation=activ, padding='same')(l4)
    l5 = BatchNormalization()(l5)
    
    l6 = Conv2D(24, 3, strides=1, dilation_rate = 32, activation=activ, padding='same')(l5)
    l6 = BatchNormalization()(l6)
    
    l7 = Conv2D(24, 3, strides=1, dilation_rate = 64, activation=activ, padding='same')(l6)
    l7 = BatchNormalization()(l7)
    
    l8 = Conv2D(24, 3, strides=1, dilation_rate = 1, activation=activ, padding='same')(l7)
    
    l9 = Conv2D(3, 1, strides=1, dilation_rate =1, activation=activ, padding='same')(l9)
    
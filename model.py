from keras.layers import Input, concatenate, Dropout, Dense, Flatten, Activation
from keras.layers.convolutional import MaxPooling2D, Conv2D, AveragePooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model

from keras import backend as K

"""
Implementation of Inception Network v4 [Inception Network v4 Paper](http://arxiv.org/pdf/1602.07261v1.pdf) in Keras.
"""


def conv_block(x, filters, nb_row, nb_col, padding='same', strides=(1, 1), bias=False):
    if K.image_dim_ordering() == "th":
        channel_axis = 1
    else:
        channel_axis = -1
    x = Conv2D(filters=filters, kernel_size=(nb_row, nb_col), padding=padding, strides=strides, bias=bias)(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)
    return x


def inception_stem(input):
    if K.image_dim_ordering() == "th":
        channel_axis = 1
    else:
        channel_axis = -1
    # Input Shape is 299 x 299 x 3 (th) or 3 x 299 x 299 (th)
    x = conv_block(input, 32, 3, 3, strides=(2, 2), padding='valid')
    x = conv_block(x, 32, 3, 3, padding='valid')
    x = conv_block(x, 64, 3, 3)

    x1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(x)
    x2 = conv_block(x, 96, 3, 3, strides=(2, 2), padding='valid')

    x = concatenate([x1, x2], axis=channel_axis)

    x1 = conv_block(x, 64, 1, 1)
    x1 = conv_block(x1, 96, 3, 3, padding='valid')

    x2 = conv_block(x, 64, 1, 1)
    x2 = conv_block(x2, 64, 1, 7)
    x2 = conv_block(x2, 64, 7, 1)
    x2 = conv_block(x2, 96, 3, 3, padding='valid')

    x = concatenate([x1, x2], axis=channel_axis)

    x1 = conv_block(x, 192, 3, 3, strides=(2, 2), padding='valid')
    x2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(x)

    x = concatenate([x1, x2], axis=channel_axis)
    return x


def inception_A(input):
    if K.image_dim_ordering() == "th":
        channel_axis = 1
    else:
        channel_axis = -1
    a1 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input)
    a1 = conv_block(a1, 96, 1, 1)

    a2 = conv_block(input, 96, 1, 1)

    a3 = conv_block(input, 64, 1, 1)
    a3 = conv_block(a3, 96, 3, 3)

    a4 = conv_block(input, 64, 1, 1)
    a4 = conv_block(a4, 96, 3, 3)
    a4 = conv_block(a4, 96, 3, 3)

    m = concatenate([a1, a2, a3, a4], axis=channel_axis)
    return m


def inception_B(input):
    if K.image_dim_ordering() == "th":
        channel_axis = 1
    else:
        channel_axis = -1
    b1 = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input)
    b1 = conv_block(b1, 128, 1, 1)

    b2 = conv_block(input, 384, 1, 1)

    b3 = conv_block(input, 192, 1, 1)
    b3 = conv_block(b3, 224, 1, 7)
    b3 = conv_block(b3, 256, 7, 1)

    b4 = conv_block(input, 192, 1, 1)
    b4 = conv_block(b4, 192, 7, 1)
    b4 = conv_block(b4, 224, 1, 7)
    b4 = conv_block(b4, 224, 7, 1)
    b4 = conv_block(b4, 256, 1, 7)

    m = concatenate([b1, b2, b3, b4], axis=channel_axis)
    return m


def inception_C(input):
    if K.image_dim_ordering() == "th":
        channel_axis = 1
    else:
        channel_axis = -1
    c1 = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input)
    c1 = conv_block(c1, 256, 1, 1)

    c2 = conv_block(input, 256, 1, 1)

    c3 = conv_block(input, 384, 1, 1)
    c3_1 = conv_block(c3, 256, 1, 3)
    c3_2 = conv_block(c3, 256, 3, 1)
    c3 = concatenate([c3_1, c3_2], axis=channel_axis)

    c4 = conv_block(input, 384, 1, 1)
    c4 = conv_block(c4, 384, 3, 1)
    c4 = conv_block(c4, 448, 1, 3)
    c4_1 = conv_block(c4, 256, 1, 3)
    c4_2 = conv_block(c4, 256, 3, 1)
    c4 = concatenate([c4_1, c4_2], axis=channel_axis)

    m = concatenate([c1, c2, c3, c4], axis=channel_axis)
    return m


def reduction_A(input):
    if K.image_dim_ordering() == "th":
        channel_axis = 1
    else:
        channel_axis = -1
    r1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(input)

    r2 = conv_block(input, 384, 3, 3, strides=(2, 2), padding='valid')

    r3 = conv_block(input, 192, 1, 1)
    r3 = conv_block(r3, 224, 3, 3)
    r3 = conv_block(r3, 256, 3, 3, strides=(2, 2), padding='valid')

    m = concatenate([r1, r2, r3], axis=channel_axis)
    return m


def reduction_B(input):
    if K.image_dim_ordering() == "th":
        channel_axis = 1
    else:
        channel_axis = -1

    r1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(input)

    r2 = conv_block(input, 192, 1, 1)
    r2 = conv_block(r2, 192, 3, 3, strides=(2, 2), padding='valid')

    r3 = conv_block(input, 256, 1, 1)
    r3 = conv_block(r3, 256, 1, 7)
    r3 = conv_block(r3, 320, 7, 1)
    r3 = conv_block(r3, 320, 3, 3, strides=(2, 2), padding='valid')
    m = concatenate([r1, r2, r3], axis=channel_axis)
    return m


def create_inception_v4(img_height=299, img_width=299, nb_classes=1001, load_weight=True):
    '''
    Creates a inception v4 network
    :param nb_classes: number of classes.txt
    :return: Keras Model with 1 input and 1 output
    '''
    if K.image_dim_ordering() == 'th':
        init = Input((1, img_height, img_width))
    else:
        init = Input((img_height, img_width, 1))
    x = inception_stem(init)

    # 4 x Inception A
    for i in range(4):
        x = inception_A(x)

    # Reduction A
    x = reduction_A(x)

    # 7 x Inception B
    for i in range(7):
        x = inception_B(x)

    # Reduction B
    x = reduction_B(x)

    # 3 x Inception C
    for i in range(3):
        x = inception_C(x)

    # Average Pooling
    x = AveragePooling2D((8, 8))(x)

    # Dropout
    x = Dropout(0.8)(x)
    x = Flatten()(x)

    # Output
    out = Dense(units=nb_classes, activation='softmax')(x)
    '''
    Dense:
        units: 正整数，输出空间维度。
    '''
    model = Model(init, out, name='Inception-v4')
    return model


if __name__ == "__main__":
    inception_v4 = create_inception_v4(load_weight=True)
    inception_v4.summary()

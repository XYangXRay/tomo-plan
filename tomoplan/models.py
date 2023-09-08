from telnetlib import KERMIT
from unittest import result
import tensorflow as tf
from tensorflow.keras import Sequential, Input, Model
from tensorflow.keras.layers import Dense, Conv2D, Conv3D, Conv3DTranspose,\
    Conv2DTranspose, Flatten, concatenate, \
    BatchNormalization, Dropout, LeakyReLU, ReLU  


def dense_norm(units, dropout, apply_batchnorm=True):
    initializer = tf.random_normal_initializer()

    result = Sequential()
    result.add(
        Dense(units,
        # activation=tf.nn.tanh, 
        use_bias=True, 
        kernel_initializer=initializer))
    result.add(Dropout(dropout))

    if apply_batchnorm:
        result.add(BatchNormalization())

    result.add(LeakyReLU())

    return result


def conv2d_norm(filters, size, strides, apply_batchnorm=True):
    initializer = tf.random_normal_initializer()

    result = Sequential()
    result.add(
        Conv2D(filters, 
        size, 
        strides=strides, 
        padding='same',
        kernel_initializer=initializer,))
        # activation=tf.nn.elu))

    if apply_batchnorm:
        result.add(BatchNormalization())

    result.add(LeakyReLU())

    return result


def dconv2d_norm(filters, size, strides, apply_dropout=False):
    initializer = tf.random_normal_initializer()

    result = Sequential()
    result.add(
        Conv2DTranspose(filters, size, strides=strides,
                               padding='same',
                               kernel_initializer=initializer,
                               use_bias=False))

    result.add(BatchNormalization())

    if apply_dropout:
        result.add(Dropout(0.2))

    result.add(LeakyReLU())

    return result

def conv3d_norm(filters, size, strides, apply_batchnorm=True):
    initializer = tf.random_normal_initializer()
    result = Sequential()
    result.add(
        Conv3D(filters, 
               size, 
               strides=strides,
               padding = 'same',
               kernel_initializer = initializer,))
            #    activation = tf.nn.elu))
    if apply_batchnorm:
        result.add(BatchNormalization())
    result.add(LeakyReLU())
    return result

def dconv3d_norm(filters, size, strides, apply_batchnorm=True):
    initializer = tf.random_normal_initializer()
    result = Sequential()
    result.add(
        Conv3DTranspose(filters, 
                         size, 
                         strides=strides,
                         padding = 'same',
                         kernel_initializer = initializer,))
                        #  activation = tf.nn.elu))
    if apply_batchnorm:
        result.add(BatchNormalization())
    result.add(LeakyReLU())
    return result

    

# def res_block():
#     initializer = tf.random_normal_initializer()

#     result = Sequential()

# def ResBlock(inputs):
    
#     x = Conv2DTranspose(128, 3, padding="same", activation="relu")(inputs)
#     x = Conv2DTranspose(128, 3, padding="same")(x)
#     x = Add()([inputs, x])
#     x = LeakyReLU()(x)
#     x = BatchNormalization()(x)
#     return x

def resblock(x, filters, size):
    x = Conv2D(filters, size, padding='same')(x)
    fx = BatchNormalization()(x)
    fx = Conv2DTranspose(filters, size, activation='relu', padding='same')(x)
    fx = BatchNormalization()(fx)
    fx = Conv2DTranspose(filters, size, padding='same')(fx)
    fx = BatchNormalization()(fx)
    out = concatenate([x,fx], axis=3)
    out = LeakyReLU()(out)
    out = BatchNormalization()(out)
    return out

def resblock_dense(x, units):
    x = Dense(units, use_bias=True)(x)
    fx = BatchNormalization()(x)
    fx = Dense(units, activation='relu')(x)
    fx = BatchNormalization()(fx)
    fx = Dense(units, activation='relu')(x)
    fx = BatchNormalization()(fx)
    out = concatenate([x,fx], axis=3)
    out = LeakyReLU()(out)
    out = BatchNormalization()(out)
    return out

def resblock_3d(x, filters, size):
    x = Conv3D(filters, size, padding='same')(x)
    fx = BatchNormalization()(x)
    fx = Conv3DTranspose(filters, size, activation='relu', padding='same')(x)
    fx = BatchNormalization()(fx)
    fx = Conv3DTranspose(filters, size, padding='same')(fx)
    fx = BatchNormalization()(fx)
    out = concatenate([x,fx], axis=4)
    out = LeakyReLU()(out)
    out = BatchNormalization()(out)
    return out


def make_generator(img_h, img_w, conv_num, conv_size, dropout, output_num):
    units = 128
    fc_size = img_w ** 2
    inputs = Input(shape=(img_h, img_w, 1))
    x = Flatten()(inputs)
    fc_stack = [
        dense_norm(units, dropout),
        dense_norm(units, dropout),
        dense_norm(units, dropout),
        dense_norm(fc_size, 0),
    ]

    conv_stack = [
        conv2d_norm(conv_num, conv_size+2, 1),
        conv2d_norm(conv_num, conv_size+2, 1),
        conv2d_norm(conv_num, conv_size, 1),

    ]

    dconv_stack = [
        dconv2d_norm(conv_num, conv_size+2, 1),
        dconv2d_norm(conv_num, conv_size+2, 1),
        dconv2d_norm(conv_num, conv_size, 1),
    ]

    last = conv2d_norm(output_num, 3, 1)

    for fc in fc_stack:
        x = fc(x)

    x = tf.reshape(x, shape=[-1, img_w, img_w, 1])
    # Convolutions
    for conv in conv_stack:
        x = conv(x)

    for dconv in dconv_stack:
        x = dconv(x)
    x = last(x)

    return Model(inputs=inputs, outputs=x)


def make_generator_3d(batch_size, img_h, img_w):
    inputs = Input(shape=[img_h, img_w, 1])
    down_stack = [
        conv2d_norm(256, 3, 1),  # (batch_size, 128, 128, 64)
        conv2d_norm(256, 3, 2),  # (batch_size, 64, 64, 128)
        conv2d_norm(512, 3, 1),  # (batch_size, 32, 32, 256)
        conv2d_norm(512, 3, 2),  # (batch_size, 16, 16, 512)
        conv2d_norm(512, 3, 1),  # (batch_size, 8, 8, 512)
        conv2d_norm(512, 3, 2),  # (batch_size, 4, 4, 512)
        conv2d_norm(512, 3, 1),  # (batch_size, 2, 2, 512)
        conv2d_norm(128, 3, 1),  # (batch_size, 1, 1, 512)
    ]
    fc_stack = [
        dense_norm(128, 0.2),
        dense_norm(256, 0.2),
        dense_norm(128, 0.2),
        dense_norm(img_h*img_w//64, 0),
    ]

    up_stack = [
        dconv2d_norm(1, 3, 1),  # (batch_size, 2, 2, 1024)
        dconv2d_norm(512, 3, 1),  # (batch_size, 4, 4, 1024)
        dconv2d_norm(512, 3, 2),  # (batch_size, 8, 8, 1024)
        dconv2d_norm(512, 3, 1),  # (batch_size, 16, 16, 1024)
        dconv2d_norm(512, 3, 2),  # (batch_size, 32, 32, 512)
        dconv2d_norm(512, 3, 1),  # (batch_size, 64, 64, 256)
        dconv2d_norm(256, 3, 2),  # (batch_size, 128, 128, 128)
        dconv2d_norm(256, 5, 1),
        dconv2d_norm(256, 5, 1),
        dconv2d_norm(256, 5, 1),
    ]
    last = conv2d_norm(128, 3, 1)    

    x = inputs

    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])
    x = tf.keras.layers.Flatten()(x)
    for fc in fc_stack:
        x = fc(x)
    x = tf.reshape(x, [batch_size, img_h//8, img_w//8, 1])


    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        # x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


def make_generator_3drec(nang, px):
    inputs = Input(shape=[px, px, nang])
    down_stack = [
        conv2d_norm(128, 3, 1),  # (batch_size, 128, 128, 64)
        conv2d_norm(128, 3, 2),  # (batch_size, 64, 64, 128)
        conv2d_norm(256, 3, 1),  # (batch_size, 32, 32, 256)
        conv2d_norm(256, 3, 2),  # (batch_size, 16, 16, 512)
        conv2d_norm(256, 3, 1),  # (batch_size, 8, 8, 512)
        conv2d_norm(256, 3, 2),  # (batch_size, 4, 4, 512)
        conv2d_norm(256, 3, 1),  # (batch_size, 2, 2, 512)
        conv2d_norm(128, 3, 1),  # (batch_size, 1, 1, 512)
    ]
    fc_stack = [
        dense_norm(128, 0.2),
        dense_norm(256, 0.2),
        dense_norm(256, 0.2),
        dense_norm(px**2//64*128, 0),
    ]

    up_stack = [
        conv2d_norm(128, 3, 1),  # (batch_size, 2, 2, 1024)
        conv2d_norm(256, 3, 1),  # (batch_size, 4, 4, 1024)
        dconv2d_norm(256, 3, 2),  # (batch_size, 8, 8, 1024)
        conv2d_norm(256, 3, 1),  # (batch_size, 16, 16, 1024)
        dconv2d_norm(256, 3, 2),  # (batch_size, 32, 32, 512)
        conv2d_norm(256, 3, 1),  # (batch_size, 64, 64, 256)
        dconv2d_norm(128, 3, 2),  # (batch_size, 128, 128, 128)
        dconv2d_norm(128, 3, 1),
        dconv2d_norm(128, 3, 1),
        dconv2d_norm(128, 3, 1),
    ]
    last = conv2d_norm(128, 3, 1)    

    x = inputs

    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])
    x = tf.keras.layers.Flatten()(x)
    for fc in fc_stack:
        x = fc(x)
    x = tf.reshape(x, [1, px//8, px//8, 128])


    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        # x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


def make_generator_conv3d(nang, px):
    inputs = Input(shape=[px, px, nang])
    down_stack = [
        conv2d_norm(128, 3, 1),  # (batch_size, 128, 128, 64)
        conv2d_norm(128, 3, 2),  # (batch_size, 64, 64, 128)
        conv2d_norm(256, 3, 1),  # (batch_size, 32, 32, 256)
        conv2d_norm(256, 3, 2),  # (batch_size, 16, 16, 512)
        conv2d_norm(256, 3, 1),  # (batch_size, 8, 8, 512)
        conv2d_norm(256, 3, 2),  # (batch_size, 4, 4, 512)
        conv2d_norm(256, 3, 1),  # (batch_size, 2, 2, 512)
        conv2d_norm(32, 3, 1),  # (batch_size, 1, 1, 512)
    ]
    fc_stack = [
        dense_norm(256, 0.2),
        dense_norm(512, 0.2),
        dense_norm(128, 0.2),
        dense_norm(px**3//512*8, 0),
    ]

    up_stack = [
        dconv3d_norm(32, 3, 1),  # (batch_size, 2, 2, 1024)
        dconv3d_norm(128, 3, 1),  # (batch_size, 4, 4, 1024)
        dconv3d_norm(128, 3, 2),  # (batch_size, 8, 8, 1024)
        dconv3d_norm(64, 3, 1),  # (batch_size, 16, 16, 1024)
        dconv3d_norm(64, 3, 2),  # (batch_size, 32, 32, 512)
        dconv3d_norm(32, 3, 1),  # (batch_size, 64, 64, 256)
        dconv3d_norm(32, 3, 2),  # (batch_size, 128, 128, 128)
        dconv3d_norm(32, 3, 1),
        dconv3d_norm(32, 3, 1),
        dconv3d_norm(16, 3, 1),
    ]
    last = conv3d_norm(1, 3, 1)    

    x = inputs

    # Downsampling through the model
    # skips = []
    for down in down_stack:
        x = down(x)
        # skips.append(x)

    # skips = reversed(skips[:-1])
    x = tf.keras.layers.Flatten()(x)
    for fc in fc_stack:
        x = fc(x)
    x = tf.reshape(x, [1, px//8, px//8, px//8, 8])


    # Upsampling and establishing the skip connections
    for up in up_stack:
        x = up(x)
        # x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)



def make_generator_conv3d1(nang, px):
    inputs = Input(shape=[px, px, nang])
    down_stack = [
        conv2d_norm(256, 3, 1),  # (batch_size, 128, 128, 64)
        conv2d_norm(256, 3, 2),  # (batch_size, 64, 64, 128)
        conv2d_norm(512, 3, 1),  # (batch_size, 32, 32, 256)
        conv2d_norm(512, 3, 2),  # (batch_size, 16, 16, 512)
        conv2d_norm(512, 3, 1),  # (batch_size, 8, 8, 512)
        conv2d_norm(512, 3, 2),  # (batch_size, 4, 4, 512)
        conv2d_norm(512, 3, 1),  # (batch_size, 2, 2, 512)
        conv2d_norm(16, 3, 1),  # (batch_size, 1, 1, 512)
    ]
    fc_stack = [
        dense_norm(16, 0.2),
        dense_norm(256, 0.2),
        dense_norm(128, 0.2),
        dense_norm(px**3//512, 0),
    ]

    up_stack = [
        dconv3d_norm(16, 3, 1),  # (batch_size, 2, 2, 1024)
        dconv3d_norm(512, 3, 1),  # (batch_size, 4, 4, 1024)
        dconv3d_norm(512, 3, 2),  # (batch_size, 8, 8, 1024)
        dconv3d_norm(512, 3, 1),  # (batch_size, 16, 16, 1024)
        dconv3d_norm(512, 3, 2),  # (batch_size, 32, 32, 512)
        dconv3d_norm(512, 3, 1),  # (batch_size, 64, 64, 256)
        dconv3d_norm(256, 3, 2),  # (batch_size, 128, 128, 128)
        dconv3d_norm(256, 3, 1),
        # dconv3d_norm(256, 3, 1),
        # dconv3d_norm(256, 3, 1),
    ]
    last = conv3d_norm(1, 3, 1)    

    x = inputs

    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        # skips.append(x)

    skips = reversed(skips[:-1])
    # x = tf.keras.layers.Flatten()(x)
    for fc in fc_stack:
        x = fc(x)
    x = tf.reshape(x, [1, px//8, px//8, px//8, 256])


    # Upsampling and establishing the skip connections
    for up in up_stack:
        x = up(x)
        # x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


def make_generator_conv3d2(nang, px):
    inputs = Input(shape=[px, px, nang])
    down_stack = [
        conv2d_norm(128, 3, 1),  # (batch_size, 128, 128, 64)
        # conv2d_norm(256, 3, 1),  # (batch_size, 64, 64, 128)
        # conv2d_norm(128, 3, 1),  # (batch_size, 32, 32, 256)
    ]
    fc_stack = [
        dense_norm(128, 0.2),
        dense_norm(128, 0.2),
        dense_norm(128, 0.2),
        dense_norm(128, 0),
    ]

    up_stack = [
        dconv3d_norm(1, 3, 1),  # (batch_size, 2, 2, 1024)
        dconv3d_norm(64, 3, 1),  # (batch_size, 4, 4, 1024)
        dconv3d_norm(64, 3, 1),  # (batch_size, 8, 8, 1024)
        dconv3d_norm(64, 3, 1),  # (batch_size, 16, 16, 1024)
        dconv3d_norm(64, 3, 1),  # (batch_size, 32, 32, 512)
        # dconv3d_norm(512, 3, 1),  # (batch_size, 64, 64, 256)
        # dconv3d_norm(256, 3, 1),  # (batch_size, 128, 128, 128)
        # dconv3d_norm(256, 3, 1),
        # dconv3d_norm(256, 3, 1),
        # dconv3d_norm(256, 3, 1),
    ]
    last = conv3d_norm(1, 3, 1)    

    x = inputs

    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        # skips.append(x)

    skips = reversed(skips[:-1])
    # x = tf.keras.layers.Flatten()(x)
    for fc in fc_stack:
        x = fc(x)
    x = tf.reshape(x, [1, px, px, px, 1])


    # Upsampling and establishing the skip connections
    for up in up_stack:
        x = up(x)
        # x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)

def make_generator_conv3dres(nang, px):
    inputs = Input(shape=[px, px, nang])
    x = inputs
    x = conv2d_norm(128, 3, 1)(x)
    x = resblock_dense(x, 128)
    x = tf.reshape(x, [1, px, px, px, 2])
    x = resblock_3d(x, 128, 3)
    x = conv3d_norm(1, 3, 1)(x)    

   

    return tf.keras.Model(inputs=inputs, outputs=x)



def make_generator_3dped1(img_h, img_w):
    inputs = Input(shape=[img_h, img_w, 1])
    x = inputs
    # ed1 = conv2d_norm(32, 3, 1)(x)
    # ed1 = conv2d_norm(32, 3, 2)(ed1)
    # ed1 = dconv2d_norm(32, 3, 2)(ed1)
    # ed1 = dconv2d_norm(32, 3, 1)(ed1)

    # ed2 = conv2d_norm(64, 3, 1)(x)
    # ed2 = conv2d_norm(64, 3, 4)(ed2)
    # ed2 = dconv2d_norm(64, 3, 4)(ed2)
    # ed2 = dconv2d_norm(64, 3, 1)(ed2)

    # ed3 = conv2d_norm(128, 3, 1)(x)
    # ed3 = conv2d_norm(128, 3, 8)(ed3)
    # ed3 = dconv2d_norm(128, 3, 8)(ed3)
    # ed3 = dconv2d_norm(128, 3, 1)(ed3)

    # ed4 = conv2d_norm(256, 3, 1)(x)
    # ed4 = conv2d_norm(256, 3, 16)(ed4)
    # ed4 = dconv2d_norm(256, 3, 16)(ed4)
    # ed4 = dconv2d_norm(256, 3, 1)(ed4)

    # x = layers.concatenate([ed1, ed2, ed3, ed4], axis=3)


    # ed2_stack = [
    #     conv2d_norm(64, 3, 1),
    #     conv2d_norm(64, 3, 4),
    #     dconv2d_norm(64, 3, 4),
    #     dconv2d_norm(64, 3, 1)
    #      ]
    # ed3_stack = [
    #     conv2d_norm(128, 3, 1),
    #     conv2d_norm(128, 3, 8),
    #     dconv2d_norm(128, 3, 8),
    #     dconv2d_norm(128, 3, 1)
    #      ]
    # ed4_stack = [
    #     conv2d_norm(256, 3, 1),
    #     conv2d_norm(256, 3, 16),
    #     dconv2d_norm(256, 3, 16),
    #     dconv2d_norm(256, 3, 1)
    #      ]
    x = resblock(x, 128, 3)
    x = resblock(x, 128, 3)
    x = resblock(x, 128, 3)
    x = resblock(x, 256, 3)
    x = resblock(x, 512, 3)
    # x = dconv2d_norm(256, 3, 1)(x)
    # x = dconv2d_norm(256, 3, 1)(x)
    # x = dconv2d_norm(256, 3, 1)(x)
    # x = dconv2d_norm(256, 3, 1)(x)
    # x = dconv2d_norm(256, 3, 1)(x)
    # x = dconv2d_norm(256, 3, 1)(x)
    # x = dconv2d_norm(256, 3, 1)(x)
    # x = dconv2d_norm(256, 3, 1)(x)
    # x = dconv2d_norm(256, 3, 1)(x)    
    # x = inputs
    # ed = []
    # for ed1, ed2, ed3, ed4 in zip(ed1_stack, ed2_stack, ed3_stack, ed4_stack):
    #     x = tf.keras.layers.Concatenate()([ed1(x), ed2(x), ed3(x), ed4(x)])

    # for ed1 in ed1_stack:
    #     ed = ed1(x)
    # for ed2 in ed2_stack:
    #     ed.append(ed2(x))
    # for ed3 in ed3_stack:
    #     ed.append(ed3(x))
    # for ed4 in ed4_stack:
    #     ed.append(ed4(x))

    # x1 = ed1(x)
    # x2 = ed2(x)
    # x3 = ed3(x)
    # x4 = ed4(x)
    # ed = tf.keras.layers.Concatenate()([x1, x2, x3, x4])
    x = conv2d_norm(128, 3, 1)(x)
    return Model(inputs=inputs, outputs=x)


def make_generator_3dped(img_h, img_w):
    inputs = Input(shape=[img_h, img_w, 1])
    ed1_stack = [
        conv2d_norm(32, 3, 1),
        conv2d_norm(32, 3, 2),
        dconv2d_norm(32, 3, 2),
        dconv2d_norm(32, 3, 1)
        ]
    ed2_stack = [
        conv2d_norm(64, 3, 1),
        conv2d_norm(64, 3, 4),
        dconv2d_norm(64, 3, 4),
        dconv2d_norm(64, 3, 1)
         ]
    ed3_stack = [
        conv2d_norm(128, 3, 1),
        conv2d_norm(128, 3, 8),
        dconv2d_norm(128, 3, 8),
        dconv2d_norm(128, 3, 1)
         ]
    ed4_stack = [
        conv2d_norm(256, 3, 1),
        conv2d_norm(256, 3, 16),
        dconv2d_norm(256, 3, 16),
        dconv2d_norm(256, 3, 1)
         ]
    dconv_stack = [
        dconv2d_norm(256, 3, 1),
        dconv2d_norm(256, 3, 1),
        dconv2d_norm(256, 3, 1),
    ]
    last = dconv2d_norm(128, 3, 1)    
    x = inputs
    x1, x2, x3, x4 = [], [], [], []
    for ed1, ed2, ed3, ed4 in zip(ed1_stack, ed2_stack, ed3_stack, ed4_stack):
        x1 = ed1(x)
        x2 = ed2(x)
        x3 = ed3(x)
        x4 = ed4(x)
    x = concatenate([x1, x2, x3, x4], axis=3)
    for dconv in dconv_stack:
        x = dconv(x)

    # for ed1 in ed1_stack:
    #     ed = ed1(x)
    # for ed2 in ed2_stack:
    #     ed.append(ed2(x))
    # for ed3 in ed3_stack:
    #     ed.append(ed3(x))
    # for ed4 in ed4_stack:
    #     ed.append(ed4(x))

    # x1 = ed1(x)
    # x2 = ed2(x)
    # x3 = ed3(x)
    # x4 = ed4(x)
    # ed = tf.keras.layers.Concatenate()([x1, x2, x3, x4])
    x = last(x)
    return tf.keras.Model(inputs=inputs, outputs=x)





def make_discriminator():
    model = tf.keras.Sequential()
    model.add(Conv2D(32, (5, 5), strides=(2, 2), padding='same'))
    model.add(Conv2D(32, (5, 5), strides=(1, 1), padding='same'))
    model.add(LeakyReLU())
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (5, 5), strides=(2, 2), padding='same'))
    model.add(Conv2D(64, (5, 5), strides=(1, 1), padding='same'))
    model.add(LeakyReLU())
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3, 3), strides=(2, 2), padding='same'))
    model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same'))
    model.add(LeakyReLU())
    model.add(Dropout(0.2))

    model.add(Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
    model.add(Conv2D(128, (3, 3), strides=(1, 1), padding='same'))
    model.add(LeakyReLU())
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(1))
    return model


def make_discriminator_3d():
    model = tf.keras.Sequential()
    model.add(Conv3D(16, (3, 3, 3), strides=(1, 1, 1), padding='same'))
    model.add(Conv3D(16, (3, 3, 3), strides=(2, 2, 2), padding='same'))
    model.add(Conv3D(16, (3, 3, 3), strides=(1, 1, 1), padding='same'))
    model.add(LeakyReLU())
    model.add(Dropout(0.2))

    
    model.add(Conv3D(32, (3, 3, 3), strides=(2, 2, 2), padding='same'))
    model.add(Conv3D(32, (3, 3, 3), strides=(1, 1, 1), padding='same'))
    model.add(LeakyReLU())
    model.add(Dropout(0.2))

    model.add(Conv3D(64, (3, 3, 3), strides=(2, 2, 2), padding='same'))
    model.add(Conv3D(64, (3, 3, 3), strides=(1, 1, 1), padding='same'))
    model.add(LeakyReLU())
    model.add(Dropout(0.2))

    model.add(Conv3D(64, (3, 3, 3), strides=(2, 2, 2), padding='same'))
    model.add(Conv3D(64, (3, 3, 3), strides=(1, 1, 1), padding='same'))
    model.add(LeakyReLU())
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(1))


    return model

import tensorflow as tf
from tensorflow.keras import layers

def double_conv_block(x, n_filters):
    """
    Double convolution block with specified number of filters.

    :param x: Input tensor.
    :param n_filters: Number of filters.
    :return: Output tensor after two convolution operations.
    """
    x = layers.Conv2D(n_filters, 3, padding="same", activation="relu", kernel_initializer='random_normal')(x)
    x = layers.Conv2D(n_filters, 3, padding="same", activation="relu", kernel_initializer='random_normal')(x)
    return x

def downsample_block(x, n_filters): 
    """
    Downsample block with double convolution, max pooling, and dropout.

    :param x: Input tensor.
    :param n_filters: Number of filters.
    :return: Feature tensor after double convolution, and downsampled tensor.
    """
    f = double_conv_block(x, n_filters)
    p = layers.MaxPool2D(2)(f)
    p = layers.Dropout(0.3)(p)
    return f, p

def upsample_block(x, conv_features, n_filters):
    """
    Upsample block with transposed convolution, concatenation, dropout, and double convolution.

    :param x: Input tensor.
    :param conv_features: Features from the corresponding downsample block.
    :param n_filters: Number of filters.
    :return: Output tensor after upsampling.
    """
    x = layers.Conv2DTranspose(n_filters, 3, 2, padding="same")(x)
    x = layers.concatenate([x, conv_features])
    x = layers.Dropout(0.3)(x)
    x = double_conv_block(x, n_filters)
    return x

def build_unet_model(IMAGE_SIZE):
    """
    Build the U-Net model for semantic segmentation.

    :return: U-Net model.
    """
    
    inputs = layers.Input(shape=(*IMAGE_SIZE, 3))

    # Downsample blocks
    f1, p1 = downsample_block(inputs, 16)
    f2, p2 = downsample_block(p1, 32)
    f3, p3 = downsample_block(p2, 64)

    # Bottleneck layer
    bottleneck = double_conv_block(p3, 128)

    # Upsample blocks
    u3 = upsample_block(bottleneck, f3, 64)
    u2 = upsample_block(u3, f2, 32)
    u1 = upsample_block(u2, f1, 16)

    # Output layer
    outputs = layers.Conv2D(1, 1, padding="same", activation="sigmoid")(u1)

    # Create and return the U-Net model
    unet_model = tf.keras.Model(inputs, outputs, name="U-Net")
    return unet_model

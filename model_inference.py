from google.colab import drive
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import load_model
from helper.loss_coeff import *
from helper.rle import *
from helper.unet import *

IMAGE_SIZE = (192, 192)
BATCH_SIZE = 64
CHECKPOINT_FILEPATH = '/content/unet.h5'
PRED_PATH = '/content/airbus_test'
pred_files_name = [f for f in os.listdir(PRED_PATH) if f.endswith('.jpg')]

def resize(img):
    """
    Resize the image to the specified target size.

    :param img: Input image.
    :return: Resized image.
    """
    
    img = tf.image.resize(img, IMAGE_SIZE, method="nearest")
    return img

def flip(img):
    """
    Flip the image horizontally or vertically with a certain probability.

    :param img: Input image.
    :return: Flipped image.
    """
    
    # Flip left-right with a probability greater than 0.5
    if np.random.uniform() > 0.5:
        img = tf.image.flip_left_right(img[np.newaxis])[0]

    # Flip up-down with a probability greater than 0.5
    if np.random.uniform() > 0.5:
        img = tf.image.flip_up_down(img[np.newaxis])[0]

    return np.array(img)

def normalization(img):
    """
    Normalize the pixel values of the image to the range [0, 1].

    :param img: Input image.
    :return: Normalized image.
    """
    
    img = img / 255.0
    return img

def load_image_pred(image):
    """
    Preprocess and augment testing images

    :param image: Input image.
    :return: Processed and augmented image
    """
    
    image = tf.cast(image, tf.float32)

    image = resize(image)
    image = normalization(image)
    image = flip(image)
    
    return image

# Generator function to load images for predicting
def dataset_generator(image_file_names, path):
    """
    Generates preprocessed image

    :param image_file_names: List of image file names.
    :param path: Path to the directory containing the images.
    :return: Yield preprocessed image
    """
    
    for image_name in image_file_names:
        image_path = os.path.join(path, image_name)

        # Load image using target size specified by IMAGE_SIZE
        image = load_img(image_path, target_size=None)


        # Apply preprocessing and augmentation to image
        image = load_image_pred(image)

        # Yield the processed image for each iteration
        yield image

def inference_unet_model(pred_batches, CHECKPOINT_FILEPATH):
    trained_model = load_model(CHECKPOINT_FILEPATH,
                               custom_objects={'bce_dice_loss': bce_dice_loss,
                                               'dice_coeff': dice_coeff})

    pred = trained_model.predict(pred_batches)

    return np.round(pred)

if __name__ == "__main__":
    pred_generator = lambda: dataset_generator(pred_files_name, PRED_PATH)
    # Load the test dataset and prepare batches for testing
    pred_dataset = tf.data.Dataset.from_generator(
        pred_generator,
        output_signature=(
            tf.TensorSpec(shape=(*IMAGE_SIZE, 3), dtype=tf.float32)
        )
    )

    pred_batches = pred_dataset.batch(BATCH_SIZE)

    # Perform inference using the U-Net model
    inference_result = inference_unet_model(pred_batches, CHECKPOINT_FILEPATH)

from google.colab import drive
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
from tensorflow.keras import layers, losses
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from helper.loss_coeff import *
from helper.rle import *
from helper.unet import *

### Mount Google Drive to access files
##from google.colab import drive
##drive.mount('/content/drive')
##
### Copy files from Google Drive to the current Colab working directory
##!cp "/content/drive/My Drive/airbus_train.zip" "airbus_train.zip"
##!cp "/content/drive/My Drive/airbus_test.zip" "airbus_test.zip"
##!cp "/content/drive/My Drive/train_ship_segmentations_v2.csv" "train_ship_segmentations_v2.csv"
##
### Create a directory named 'airbus' and unzip the train_data into it
##!mkdir airbus
##!unzip airbus_train.zip -d airbus
##
### Create a directory named 'airbus_test' and unzip the test_data into it
##!mkdir airbus_test
##!unzip airbus_test.zip -d airbus_test


# Reading a csv file that contains train_ship_segmentations
df = pd.read_csv('train_ship_segmentations_v2.csv')

#Define local variables
TRAIN_PATH = '/content/airbus'
TEST_PATH = '/content/airbus_test'
CHECKPOINT_FILEPATH = '/content/unet.h5'
SAVE_PATH = '/content/trained_model.h5'

BATCH_SIZE = 128
IMAGE_SIZE = (192, 192)
ORIG_IMAGE_SIZE = (768, 768)
BUFFER_SIZE = 500
NUM_EPOCHS = 100
VAL_SUBSPLITS = 5
METRIC = 'val_loss'

train_files_name = [f for f in os.listdir(TRAIN_PATH) if f.endswith('.jpg')]
test_files_name = [f for f in os.listdir(TEST_PATH) if f.endswith('.jpg')]

TRAIN_LENGTH = len(train_files_name)
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE

VALIDATION_LENGTH = int(len(test_files_name) * 0.7)
VALIDATION_STEPS = VALIDATION_LENGTH // BATCH_SIZE // VAL_SUBSPLITS

# Create masks with all ships
df['EncodedPixels'] = df['EncodedPixels'].fillna('')
all_masks = df.groupby(by='ImageId')['EncodedPixels'].agg(lambda seq: ' '.join(seq))

def resize(img, mask):
    """
    Resize the image and mask to the specified target size.

    :param img: Input image.
    :param mask: Input mask.
    :return: Resized image and mask.
    """
    
    img = tf.image.resize(img, IMAGE_SIZE, method="nearest")
    mask = tf.image.resize(mask, IMAGE_SIZE, method="nearest")
    
    return img, mask

def flip(img, mask):
    """
    Flip the image and mask horizontally or vertically with a certain probability.

    :param img: Input image.
    :param mask: Input mask.
    :return: Flipped image and mask.
    """
    
    # Flip left-right with a probability greater than 0.5
    if np.random.uniform() > 0.5:
        img = tf.image.flip_left_right(img[np.newaxis])[0]
        mask = tf.image.flip_left_right(mask[np.newaxis])[0]

    # Flip up-down with a probability greater than 0.5
    if np.random.uniform() > 0.5:
        img = tf.image.flip_up_down(img[np.newaxis])[0]
        mask = tf.image.flip_up_down(mask[np.newaxis])[0]

    return np.array(img), np.array(mask)

def normalization(img):
    """
    Normalize the pixel values of the image to the range [0, 1].

    :param img: Input image.
    :return: Normalized image.
    """
    
    img = img / 255.0
    return img

def load_image_train(image, mask):
    """
    Preprocess and augment training images and masks.

    :param image: Input image.
    :param mask: Input mask.
    :return: Processed and augmented image and mask.
    """
    
    image = tf.cast(image, tf.float32)
    mask = tf.cast(mask, tf.float32)

    image, mask = resize(image, mask)
    image = normalization(image)
    image, mask = flip(image, mask)
    
    return image, mask

# Generator function to load images and masks for training or testing
def dataset_generator(image_file_names, path, all_masks):
    """
    Generates preprocessed image and mask pairs for a given set of image file names.

    :param image_file_names: List of image file names.
    :param path: Path to the directory containing the images.
    :return: Yield preprocessed image and mask pairs.
    """
    
    for image_name in image_file_names:
        image_path = os.path.join(path, image_name)

        # Load image using target size specified by IMAGE_SIZE
        image = load_img(image_path, target_size=None)

        # Decode mask using the run-length encoding (RLE) and original image size
        mask = rle_decode(all_masks.loc[image_name], ORIG_IMG_SIZE)

        # Apply preprocessing and augmentation to image and mask
        image, mask = load_image_train(image, mask)

        # Yield the processed image and mask for each iteration
        yield image, mask
        
def train_unet_model(train_batches, validation_batches, NUM_EPOCHS, STEPS_PER_EPOCH, VALIDATION_STEPS, CHECKPOINT_FILEPATH):
    # Create an instance of the U-Net model
    unet_model = build_unet_model(IMAGE_SIZE)

    # Compile the model with Adam optimizer, BCE + Dice loss, and Dice coefficient as a metric
    unet_model.compile(optimizer='adam',
                       loss=bce_dice_loss,
                       metrics=[dice_coeff])

    # Define a ModelCheckpoint callback to save the best model during training
    checkpoint = ModelCheckpoint(
        filepath=CHECKPOINT_FILEPATH,
        monitor=METRIC,
        mode='min',
        save_best_only=True
    )

    # Define an EarlyStopping callback to stop training early if the loss doesn't improve
    earlystop = EarlyStopping(monitor='loss', patience=5)

    # List of callbacks to be used during training
    callbacks_list = [checkpoint, earlystop]

    # Train the model using the training and validation datasets
    model_history = unet_model.fit(train_batches,
                                   validation_data=validation_batches,
                                   epochs=NUM_EPOCHS,
                                   steps_per_epoch=STEPS_PER_EPOCH,
                                   validation_steps=VALIDATION_STEPS,
                                   callbacks=callbacks_list)

    # Return the trained model
    return unet_model, model_history

if __name__ == "__main__":
    train_generator = lambda: dataset_generator(train_files_name, TRAIN_PATH, all_masks)
    test_generator = lambda: dataset_generator(test_files_name, TEST_PATH, all_masks)

    # Load datasets and prepare batches for training and validation
    train_dataset = tf.data.Dataset.from_generator(
        train_generator,
        output_signature=(
            tf.TensorSpec(shape=(*IMAGE_SIZE, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(*IMAGE_SIZE, 1), dtype=tf.float32),
        )
    )

    test_dataset = tf.data.Dataset.from_generator(
        test_generator,
        output_signature=(
            tf.TensorSpec(shape=(*IMAGE_SIZE, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(*IMAGE_SIZE, 1), dtype=tf.float32),
        )
    )

    train_batches = train_dataset.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
    train_batches = train_batches.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    validation_batches = test_dataset.take(VALIDATION_LENGTH).batch(BATCH_SIZE)


    # Train the U-Net model
    trained_model, model_history = train_unet_model(train_batches, 
                                                    validation_batches, 
                                                    NUM_EPOCHS, 
                                                    STEPS_PER_EPOCH, 
                                                    VALIDATION_STEPS, 
                                                    CHECKPOINT_FILEPATH)

    # Save the trained model
    trained_model.save(SAVE_PATH)

import tensorflow as tf
from tensorflow.keras import losses

def dice_coeff(y_true, y_pred, smooth=1e-6):
    """
    Compute the dice coefficient between the true and predicted binary masks.

    :param y_true: True binary mask.
    :param y_pred: Predicted binary mask.
    :param smooth: Smoothing factor to avoid division by zero.
    :return: Dice coefficient.
    """
    
    y_true_f = tf.cast(y_true, tf.float32)
    y_pred_f = tf.cast(y_pred, tf.float32)

    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

    return score

def dice_loss(y_true, y_pred):
    """
    Compute the dice loss between the true and predicted binary masks.

    :param y_true: True binary mask.
    :param y_pred: Predicted binary mask.
    :return: Dice loss.
    """
    
    return 1 - dice_coeff(y_true, y_pred)

def bce_dice_loss(y_true, y_pred):
    """
    Compute the combination of binary crossentropy (BCE) and Dice loss.

    :param y_true: True binary mask.
    :param y_pred: Predicted binary mask.
    :return: Combined loss.
    """ 

    # Combine binary crossentropy and dice loss
    loss = losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)

    return loss

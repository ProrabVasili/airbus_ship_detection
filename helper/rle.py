import numpy as np

def rle_decode(mask_rle, shape):
    """
    Decode a run-length encoded (RLE) mask and return the corresponding binary mask.

    :param mask_rle: Run-length encoded string representing the mask.
    :param shape: Tuple representing the shape of the target binary mask (height, width).

    :return: Binary mask as a NumPy array.
    """
    # Initialize an array of zeros with the shape of the target binary mask
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)

    # Check if the input is a non-empty string
    if isinstance(mask_rle, str):
        # Split the RLE string into starts and lengths
        s = mask_rle.split()
        starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]

        # Adjust starts to zero-based indexing
        starts -= 1

        # Calculate ends based on starts and lengths
        ends = starts + lengths

        # Set the corresponding pixels in the array to 1 based on RLE information
        for lo, hi in zip(starts, ends):
            img[lo:hi] = 1

    # Reshape the 1D array to the specified shape and transpose
    img = img.reshape(shape).T 

    # Add an extra dimension to represent the channel (usually for grayscale images)
    return np.expand_dims(img, axis=-1)

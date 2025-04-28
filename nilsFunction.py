import numpy as np
import matplotlib.pyplot as plt
import math

def display_images_from_rows(array, cmap='gray'):
    """
    Display each row of a 2D numpy array as a square image.

    Parameters:
    - array: np.ndarray, shape (n_images, image_size), each row is a flattened square image.
    - cmap: colormap to use for displaying the images (default: 'gray')
    """
    if array.ndim == 1:
        array = array[np.newaxis, :]  # shape becomes (1, image_size)
    num_images = array.shape[0]
    print(f"Number of images: {num_images}")
    for i in range(num_images):
        img_flat = array[i]
        side = int(math.sqrt(len(img_flat)))
        if side * side != len(img_flat):
            raise ValueError(f"Row {i} cannot be reshaped into a square image.")
        
        img = img_flat.reshape((side, side)).T
        plt.figure()
        plt.imshow(img, cmap=cmap, vmin=0, vmax=255)
        plt.title(f"Image {i}")
        plt.axis('off')
    plt.show()
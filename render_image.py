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
    num_images = array.shape[0]
    
    for i in range(num_images):
        img_flat = array[i]
        side = int(math.sqrt(len(img_flat)))
        if side * side != len(img_flat):
            raise ValueError(f"Row {i} cannot be reshaped into a square image.")
        
        img = img_flat.reshape((side, side))
        plt.figure()
        plt.imshow(img, cmap=cmap, vmin=0, vmax=255)
        plt.title(f"Image {i}")
        plt.axis('off')
    plt.show()
# Suppose you have 3 images, each 16x16 = 256 pixels
example_data = np.random.randint(0, 256, size=(3, 256), dtype=np.uint8)
import numpy as np

    # Another pattern: an 'X'
images = np.array([[255, 0, 0, 0, 0, 0, 0, 255,0, 255, 0, 0, 0, 0, 255, 0,0, 0, 255, 0, 0, 255, 0, 0,0, 0, 0, 255, 255, 0, 0, 0,0, 0, 0, 255, 255, 0, 0, 0,0, 0, 255, 0, 0, 255, 0, 0,0, 255, 0, 0, 0, 0, 255, 0,255, 0, 0, 0, 0, 0, 0, 255],[0, 0, 255, 255, 255, 255, 0, 0,0, 255, 0, 0, 0, 0, 255, 0,255, 0, 255, 0, 0, 255, 0, 255,255, 0, 0, 0, 0, 0, 0, 255,255, 0, 255, 0, 0, 255, 0, 255,255, 0, 0, 0, 0, 0, 0, 255,0, 255, 0, 255, 255, 0, 255, 0,0, 0, 255, 255, 255, 255, 0, 0]])


display_images_from_rows(images)

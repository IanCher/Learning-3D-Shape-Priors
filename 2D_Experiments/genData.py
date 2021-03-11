import numpy as np
import os.path

from skimage.util import random_noise

from PIL import Image
from scipy import ndimage

import generate_shapes as gs


def create_datacost(img, colors):
    """ Create and initialize a datacost """

    #Extract parameters
    img_size = img.shape
    nrows = img_size[0]
    ncols = img_size[1]
    num_classes = colors.shape[0]

    #Create values of the datacost array
    values = np.zeros([nrows, ncols, num_classes], dtype=np.float32)

    for i in range(num_classes):

        sqdiff = img - colors[i,:]
        sqdiff = np.multiply(sqdiff, sqdiff)
        sqdiff = np.sum(sqdiff, axis=2, keepdims=False)

        values[:, :, i] = sqdiff
    
    values /= np.max(values)

    # Find regions where no color
    nocolor = np.sum(img, axis=2, dtype=np.uint8)
    nocolor = np.where(nocolor==0)
    values[nocolor] = 0


    return values



def main():
    """ Main function """

    print('Hello World!')

    # Parameters settings
    colors = np.array(
        [
            [255, 255, 255],
            [128, 128, 128],
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255]
        ], dtype=np.uint8
    )

    image_shape = [100, 150]
    max_shapes = 5
    min_size = 10
    maxgroundheight = 40
    max_size = 50
    gauss_noise_var = 0.05
    sp_noise_amount = 0.3

    # Create images
    i = 0
    while i < 3000:

        print("Processing {}".format(i))

        try:
            labels, images, noisy_img = gs.generate_data(
                image_shape, max_shapes=max_shapes, 
                min_size=min_size, maxgroundheight=maxgroundheight,
                max_size=max_size, allow_overlap=False,
                gauss_noise_var=gauss_noise_var, sp_noise_amount=sp_noise_amount
            )
        except (RuntimeError, TypeError, NameError, IndexError, ValueError):
            continue

        noisy_img *= 255

        # Create the datacost
        datacost = create_datacost(noisy_img, colors)

        # Convert datacost with one hot encoding
        labels_onehot = np.zeros((labels.size, 5), dtype=np.uint8)
        labels_onehot[np.arange(labels.size), labels.flatten()] = 1
        labels_onehot = np.reshape(labels_onehot, [image_shape[0], image_shape[1], 5])

        # Save images for visualization
        I = Image.fromarray(images.astype(np.uint8))
        I.save("data/synth_{}.png".format(i))

        I = Image.fromarray((noisy_img).astype(np.uint8))
        I.save("data/synth_{}_noise.png".format(i))

        I = Image.fromarray(63*labels.astype(np.uint8))
        I.save("data/synth_{}_ver.png".format(i))

        # Save groundtruth and datacost
        np.savez("data/synth_{}".format(i), groundtruth=labels_onehot, datacost=datacost)

        i += 1


if __name__ == '__main__':
    main()

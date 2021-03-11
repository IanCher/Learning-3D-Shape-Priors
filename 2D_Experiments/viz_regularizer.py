import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os.path

from PIL import Image
from scipy import ndimage

from skimage.draw import draw
from skimage._shared.utils import warn
from skimage.util import random_noise
from numpy import linalg as linalg


def conv2d(value, weights, padding="SYMMETRIC"):
    """ 2d convolution - modularize code """
    return tf.nn.conv2d(value, weights, strides=[1, 1, 1, 1], padding="SAME")


def weight_variable_const(name, value):
    """ Create and initialize weights variable """
    return tf.get_variable(
        name, dtype=tf.float32,
        initializer=value,
    )


def weight_grad_variable(name, shape):
    """ Create and initialize weights variable """

    nclasses = shape[2]

    weight_value = np.zeros([2, 2, nclasses, nclasses, 2], dtype=np.float32)
    for c in range(nclasses):
        weight_value[0, 0, c, c, 0] = -1
        weight_value[0, 1, c, c, 0] = 1

        weight_value[0, 0, c, c, 1] = -1
        weight_value[1, 0, c, c, 1] = 1

    weight_value = np.reshape(weight_value, [2, 2, nclasses, nclasses * 2])

    return tf.get_variable(name, dtype=tf.float32, initializer=weight_value)



def main ():
    """ Main function """

    # List of classes
    classes = [
        "freespace",
        "ground",
        "building",
        "vegetation",
        "roof"
    ]

    # Load the weights
    # weights_file = "weights_feat_4.npz"
    weights_file = "single_layer_W_0.npz"
    # weights_file = "multi_layer_enc_W_0.npz"
    # weights_file = "weights_shape_33.npz"
    weights = np.load(weights_file)
    weights = weights["W_1"]
    # weights = weights["W"]
    filt_y, filt_x, nclasses, _ = np.shape(weights)

    weights_tf = weight_variable_const("w", weights)

    # Sample a line along the unit circle
    nsamples = 200
    angle = np.linspace(0, 2*np.pi, nsamples)
    x_set = np.cos(angle)
    y_set = np.sin(angle)

    # Create an image with line
    image = np.ones([nsamples,2,2], dtype=np.float32)

    # Save images
    img_fold = "Img_temp"
    if not os.path.exists(img_fold):
        os.mkdir(img_fold)

    # Save figures
    fig_fold = "Figures"
    if not os.path.exists(fig_fold):
        os.mkdir(fig_fold)

    # Draw a line
    for samp in range(nsamples):
        point = np.array([x_set[samp], y_set[samp]])

        # Create the image
        for i in [-1, 1]:
            for j in [-1, 1]:
                pix  = np.array([j, i])

                pix_to_line = np.cross(point, pix)
                dist = linalg.norm(pix_to_line) / linalg.norm(point)
                val  = max(0, 1-dist)

                if pix_to_line>0:
                    image[samp, 0 if i == -1 else 1,
                                1 if j == -1 else 0] = max(val, min(1, dist))
                else:
                    image[samp, 0 if i == -1 else 1,
                                1 if j == -1 else 0] = val

        image[samp] = np.rot90(image[samp], k=-1)
                # if pix_to_line>0:
                #     image[samp, 1+i, 1-j] = max(val, min(1, dist))
                # else:
                #     image[samp, 1+i, 1-j] = val

    # Apply the convolution
    probs_tf = tf.placeholder(dtype=tf.float32, shape=[None, 2, 2, nclasses])
    res_tf = conv2d(probs_tf, weights_tf)

    # plt.style.use("ggplot")

    f, ax = plt.subplots(5, 5, figsize=(6, 6),
                         subplot_kw=dict(polar=True))


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for c in range(nclasses):

            if not os.path.exists("{}/{}".format(fig_fold, classes[c])):
                os.mkdir("{}/{}".format(fig_fold, classes[c]))

            print("{}/{}".format(fig_fold, classes[c]))

            for d in range(nclasses):

                # Prepare the probabilites
                probs = np.zeros([nsamples, 2, 2, nclasses], dtype=np.float32)
                if c == d:
                    probs[:, :, :, c] = 1
                else:
                    probs[:, :, :, c] = image
                    probs[:, :, :, d] = 1 - image

                res = sess.run([
                    res_tf
                    ],
                    feed_dict={
                        probs_tf: probs
                    }
                )

                res = res[0]
                res_x = np.abs(res[:,0,0,:nclasses])
                res_y = np.abs(res[:,0,0,nclasses:])

                res_vis = res_x + res_y

                # Save results
                res_vis = np.sum(res_vis, axis=1)

                ax[c, d].scatter(angle, res_vis,
                                 c=plt.cm.jet(res_vis/7), s=8,
                                 edgecolor=None, linewidths=0.5, zorder=1000)
                # ax[c, d].set_rmax(6)

                if c == nclasses - 1:
                    ax[c, d].set_xlabel(classes[d])
                if d == 0:
                    ax[c, d].set_ylabel(classes[c])

                if c == 0 and d == nclasses - 1:
                    # ax[c, d].set_yticklabels([2, 4, 6])
                    # ax[c, d].set_rlabel_position(20)
                    ax[c, d].set_thetagrids(np.arange(0, 360, 45),
                                            ["0°", "", "90°", "",
                                            "", "", ""],
                                            frac=1.2)
                else:
                    ax[c, d].set_xticklabels([])

                ax[c, d].set_yticks([])

        # plt.colorbar()

        plt.tight_layout(h_pad=0.1, w_pad=0.1)

        plt.savefig("plot.pdf")


if __name__ == "__main__":
    main()
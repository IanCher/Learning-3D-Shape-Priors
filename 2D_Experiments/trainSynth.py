import tensorflow as tf
import numpy as np
import os.path

from skimage.util import random_noise

from PIL import Image
from scipy import ndimage

from tensorflow.examples.tutorials.mnist import input_data



def weight_variable(name, shape):
    """ Create and initialize weights variable """
    return tf.get_variable(name, shape, dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.001))



def weight_grad_variable(name, shape):
    """ Create and initialize weights variable """

    nclasses = shape[2]

    weight_value = np.zeros([2, 2, nclasses, nclasses, 2], dtype=np.float32)
    for c in range(nclasses):
        weight_value[0, 0, c, c, 0] = -1.
        weight_value[0, 1, c, c, 0] = 1.

        weight_value[0, 0, c, c, 1] = -1.
        weight_value[1, 0, c, c, 1] = 1.

    weight_value = np.reshape(weight_value, [2, 2, nclasses, nclasses * 2])

    return tf.get_variable(name, shape, dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.001))



def conv2d(value, weights, padding="SYMMETRIC"):
    """ 2d convolution - modularize code """
    return tf.nn.conv2d(value, weights, strides=[1, 1, 1, 1], padding="SAME")



def conv2d_adj(value, weights, output_shape, padding="SYMMETRIC"):
    """ Transpose of 2 convolution + padding """
    return tf.nn.conv2d_transpose(value, weights, output_shape=output_shape, 
                                    strides=[1, 1, 1, 1], padding="SAME")



def update_lagrangian(u, l, sig):
    """ Test function """

    with tf.name_scope("Lagrange_update"):
        # Dual operations - l
        sum_u = tf.reduce_sum(u, axis=3, keep_dims=False)

        return l + sig*(sum_u - 1.0)



def update_dual(u, m, w_shape, sig):
    """ Test function """

    with tf.name_scope("Dual_update"):
        with tf.variable_scope("Weights", reuse=True):
            W = weight_variable("W", w_shape)
           
        _, _, _, nclasses = u.get_shape().as_list()
        shape_u = tf.shape(u)
        batch_size = shape_u[0]
        nrows = shape_u[1]
        ncols = shape_u[2]

        # Dual operations - m
        grad_u = conv2d(u, W)

        m += sig*grad_u

        m_rshp = tf.reshape(m, [batch_size, nrows, ncols, nclasses, 2])

        norm_m = tf.norm(m_rshp, ord='euclidean', axis=4, keep_dims=True)
        norm_m = tf.maximum(norm_m, 1.)

        m_normalize = tf.divide(m_rshp, tf.tile(norm_m, [1, 1, 1, 1, 2]))

        return  tf.reshape(m_normalize, [batch_size, nrows, ncols, nclasses * 2])



def update_primal(u, m, l, w_shape, f, tau):
    """ Test function """

    with tf.name_scope("Primal_update"):
        with tf.variable_scope("Weights", reuse=True):
            W = weight_variable("W", w_shape)

        _, _, _, nclasses = u.get_shape().as_list()
        shape_u = tf.shape(u)
        batch_size = shape_u[0]
        nrows = shape_u[1]
        ncols = shape_u[2]

        # Primal operations - u
        div_m = conv2d_adj(m, W, [batch_size, nrows, ncols, nclasses])

        u -= tau*(f + tf.tile(tf.reshape(l, [batch_size, nrows, ncols, 1]), [1, 1, 1, nclasses]) + div_m)
        u = tf.minimum(tf.cast(1, tf.float32), tf.maximum(u, tf.cast(0, tf.float32)))

        return u



def primal_dual(u, u_, m, l, w_shape, f, lam, sig, tau):
    """ Primal Dual optimization """

    with tf.name_scope("Primal_Dual"):
        # Dual update - m
        m = update_dual(u_, m, w_shape, sig)

        # Lagrangian dual update - l
        l = update_lagrangian(u_, l, sig)

        # Primal update - u
        u_0 = u
        u = update_primal(u, m, l, w_shape, f, tau)

        # Extension step
        u_ = tf.constant(2.0, name="2") *u - u_0

        return u, u_, m, l



def classification_accuracy(y_true, y_pred):
    with tf.name_scope("Accuracy"):
        return tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_true, axis=-1),
                                            tf.argmax(y_pred, axis=-1)), dtype=tf.float32))



def categorical_crossentropy(y_true, y_pred):
    """Manual computation of crossentropy"""
    _EPSILON = 10e-8

    def epsilon():
        """Returns the value of the fuzz factor used in numeric expressions"""
        return _EPSILON

    def _to_tensor(x, dtype):
        """Convert the input `x` to a tensor of type `dtype`"""
        x = tf.convert_to_tensor(x)
        if x.dtype != dtype:
            x = tf.cast(x, dtype)
        return x

    with tf.name_scope("Categorical_Cross_Entropy"):
        _epsilon = _to_tensor(epsilon(), y_pred.dtype.base_dtype)

        y_pred = tf.clip_by_value(y_pred, _epsilon, 1. - _epsilon)
        cross_entropy = -tf.reduce_sum(y_true * tf.log(y_pred))

        return cross_entropy


def build_model(params):

    nrows      = params["nrows"]
    ncols      = params["ncols"]
    nclasses   = params["nclasses"]
    batch_size = params["batch_size"]

    # Seed for reproducability
    tf.set_random_seed(1234)    

    # Feature placeholder
    features = tf.placeholder(tf.float32, [None, None, None, nclasses], name="datacost")

    # Initialize the primal and dual variables
    u  = tf.placeholder(tf.float32, [None, None, None, nclasses], name="u")
    u_ = tf.placeholder(tf.float32, [None, None, None, nclasses], name="u_")
    m  = tf.placeholder(tf.float32, [None, None, None, nclasses * 2], name="m")
    l  = tf.placeholder(tf.float32, [None, None, None], name="l")

    # Load the weight matrix (to be learned)
    w_shape = params["w_shape"]

    with tf.variable_scope("Weights") as scope:
        weight_grad_variable("W", w_shape)

    # Load the parameters
    sig = params["sig"]
    tau = params["tau"]
    lam = params["lam"]
    niter = params["niter"]

    features_lam = tf.multiply(features, tf.constant(lam, name="Lam"), name="datacost_lam")
    u_loop = u
    u_loop_= u_
    m_loop = m
    l_loop = l

    for _ in range(niter):
        u_loop, u_loop_, m_loop, l_loop = primal_dual(
            u_loop, u_loop_, m_loop, l_loop, w_shape, features_lam, lam, sig, tau
        )

    probs = u_loop

    return probs, features, u, u_, m, l



def train_model(features_data, labels_data, params):
    """ Train the model """

    # Load the hyper parameters
    nepochs = params["nepochs"]
    batch_size_max = params["batch_size"]
    ndata = params["ndata"]
    nrows = params["nrows"]
    ncols = params["ncols"]
    nclasses = params["nclasses"]

    nbatch = np.ceil(ndata/batch_size_max)

    nimg_show = params["nimg_show"]
    colors = params["colors"]

    # Build the model
    probs, features, u, u_, m, l = build_model(params)
    labels = tf.placeholder(tf.float32, probs.shape, name="prob_GT")

    # Create loss and accuracy operator
    loss_op = categorical_crossentropy(labels, probs)
    accuracy_op = classification_accuracy(labels, probs)

    # Create summaries for tensorboard visualization
    loss_mean_tf = tf.placeholder(tf.float32, name="loss_summary")
    acc_mean_tf  = tf.placeholder(tf.float32, name="acc_summary")
    seg_imgs_tf  = tf.placeholder(tf.float32, [None, None, None, 3], name="seg_img")
    prob_imgs_tf = tf.placeholder(tf.float32, [None, None, None, 3], name="prob_img")
    gt_imgs_tf  = tf.placeholder(tf.float32, [None, None, None, 3], name="gt_img")

    tf.summary.scalar('cce_loss'     , loss_mean_tf)
    tf.summary.scalar('accuracy'     , acc_mean_tf )
    tf.summary.image ('segmentation' , seg_imgs_tf, nimg_show)
    tf.summary.image ('probabilities', prob_imgs_tf, nimg_show)
    tf.summary.image ('GroundTruth', gt_imgs_tf, nimg_show)
    merged_summary = tf.summary.merge_all()

    # Create trainig operator
    learning_rate = params['learning_rate']
    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_op)

    # Initialize log file
    log_file = open("logg.txt", "w")
    log_file.close()

    with tf.Session() as sess:

        writer = tf.summary.FileWriter("Visualization")
        writer.add_graph(sess.graph)
        sess.run(tf.global_variables_initializer())

        idxs = np.arange(ndata)

        for epoch in range(nepochs):
            np.random.shuffle(idxs)

            loss_mean = 0
            acc_mean = 0

            for b_start in range(0, ndata, batch_size_max):
                b_end = min(b_start + batch_size_max, ndata)
                batch_idxs = idxs[b_start:b_end]
                batch_size = b_end - b_start

                u_init  = 1.0/nclasses * np.ones([batch_size, nrows, ncols, nclasses],  dtype=np.float32)
                u_init_ = 1.0/nclasses * np.ones([batch_size, nrows, ncols, nclasses],  dtype=np.float32)
                m_init  = np.zeros([batch_size, nrows, ncols, nclasses*2],  dtype=np.float32)
                l_init  = np.zeros([batch_size, nrows, ncols],  dtype=np.float32)

                features_batch = features_data[batch_idxs]
                labels_batch   = labels_data[batch_idxs]

                _, loss, accuracy = sess.run(
                    [train_op, loss_op, accuracy_op], 
                    feed_dict={
                        features: features_batch,
                        labels: labels_batch,
                        u: u_init,
                        u_:u_init_,
                        l:l_init,
                        m:m_init
                    }
                )

                loss_mean += loss
                acc_mean  += accuracy

                print("\n=============================================================\n")
                print("Epoch: {}, Batch: {} -- Loss: {}, Accuracy: {}\n".format(epoch, b_start, loss, accuracy))

            # Create the Weights to print in the log file    
            with tf.variable_scope("Weights", reuse=True):
                W_value = sess.run(tf.get_variable("W"))

            # Create the mean loss and accuracy
            loss_mean /= nbatch
            acc_mean /= nbatch

            # Show some results obtained with a forward pass
            u_init  = 1.0/nclasses * np.ones([nimg_show, nrows, ncols, nclasses],  dtype=np.float32)
            u_init_ = 1.0/nclasses * np.ones([nimg_show, nrows, ncols, nclasses],  dtype=np.float32)
            m_init  = np.zeros([nimg_show, nrows, ncols, nclasses*2],  dtype=np.float32)
            l_init  = np.zeros([nimg_show, nrows, ncols],  dtype=np.float32)

            seg_ex = sess.run(
                probs,
                feed_dict={
                    features: features_data[0:nimg_show],
                    u: u_init,
                    u_:u_init_,
                    l:l_init,
                    m:m_init
                }
            )

            probvis = np.zeros([nimg_show, nrows, ncols, 3], dtype=np.float32)
            segvis  = np.zeros([nimg_show, nrows, ncols, 3], dtype=np.float32)
            gtvis   = np.zeros([nimg_show, nrows, ncols, 3], dtype=np.uint8)

            for im in range(nimg_show):
                for i in range(nclasses):
                    for c in range(3):
                        probvis[im, :, :, c] += colors[i,c] * seg_ex[im, :, :, i]
                        gtvis[im, :, :, c]   += colors[i,c] * labels_data[im, :, :, i]

                best_label = np.argmax(seg_ex[im, :, :, :], axis=-1)
                segvis[im, :, :, :] = np.take(colors, best_label, axis=0)
                        
            # Write the summaries to disk
            s = sess.run(
                merged_summary, 
                feed_dict={
                    loss_mean_tf: loss_mean, 
                    acc_mean_tf: acc_mean,
                    seg_imgs_tf: segvis,
                    prob_imgs_tf: probvis,
                    gt_imgs_tf: gtvis
                    }
                )
            writer.add_summary(s, epoch)

            # Write the log file
            with open("logg.txt", "a") as log_file:
                log_file.write("\n=============================================================\n")
                log_file.write("Iter: {} -- CCE Loss: {} -- Accuracy: {}\n\n".format(epoch, loss_mean, acc_mean))
                log_file.write("Weights: \n{}\n\n".format(np.array_str(W_value)))



def main():
    """ Create data for training """

    # Data parameters
    ndata = 3000
    nrows = 100
    ncols = 150
    nclasses = 5

    colors = np.array(
        [
            [255, 255, 255],
            [128, 128, 128],
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255]
        ], dtype=np.uint8
    )

    # Load the data
    datacosts    = ndata*[None]
    groundtruths = ndata*[None]

    print("Loading data")
    for i in range(ndata):
        data = np.load("data/synth_{}.npz".format(i))

        datacosts[i] = data['datacost']
        groundtruths[i] = data['groundtruth']

    print("Data Loaded")
    
    datacosts    = np.array(datacosts)
    groundtruths = np.array(groundtruths)

    # Create parameters
    params = {
        'ndata': ndata,
        'nepochs': 3000,
        'batch_size': 32,
        'nclasses': nclasses,
        'nrows': nrows,
        'ncols': ncols,
        'w_shape': [2, 2, nclasses, 2*nclasses],
        'niter': 50,
        'sig': 0.2,
        'tau': 0.2,
        'lam': 1.0,
        'learning_rate': 0.001,
        'nimg_show': 10,
        'colors': colors
    }

    # Train the model
    tf.logging.set_verbosity(tf.logging.INFO)

    np.random.seed(0)
    train_model(datacosts, groundtruths, params)

 
if __name__ == '__main__':
    main()
import tensorflow as tf
import primal_dual as pd 
from tf_ops import *
from multiscale import avg_pool3d



def build_model(params):
    nlevels = params["nlevels"]
    nrows = params["nrows"]
    ncols = params["ncols"]
    nslices = params["nslices"]
    nclasses = params["nclasses"]
    softmax_scale = params["softmax_scale"]

    # Setup placeholders and variables.

    d = tf.placeholder(tf.float32, [None, nrows, ncols,
                                    nslices, nclasses], name="d")

    u = []
    u_ = []
    m = []
    l = []
    for level in range(nlevels):
        factor = 2 ** level
        assert nrows % factor == 0
        assert ncols % factor == 0
        assert nslices % factor == 0
        nrows_level = nrows // factor
        ncols_level = ncols // factor
        nslices_level = nslices // factor
        u.append(tf.placeholder(
                     tf.float32, [None, nrows_level, ncols_level,
                                  nslices_level, nclasses],
                     name="u{}".format(level)))
        u_.append(tf.placeholder(
                      tf.float32, [None, nrows_level, ncols_level,
                                   nslices_level, nclasses],
                     name="u_{}".format(level)))
        m.append(tf.placeholder(
                     tf.float32, [None, nrows_level, ncols_level,
                                  nslices_level, 3 * nclasses],
                     name="m{}".format(level)))
        l.append(tf.placeholder(
                     tf.float32, [None, nrows_level, ncols_level,
                                  nslices_level],
                     name="l{}".format(level)))

        with tf.variable_scope("weights{}".format(level)):
            conv_weight_variable(
                "w1", [2, 2, 2, nclasses, 3 * nclasses], stddev=0.001)
            conv_weight_variable(
                "w2", [2, 2, 2, nclasses, 3 * nclasses], stddev=0.001)

    sig = params["sig"]
    tau = params["tau"]
    lam = params["lam"]
    niter = params["niter"]

    d_lam = tf.multiply(d, lam, name="d_lam")

    d_encoded = []
    for level in range(nlevels):
        with tf.name_scope("datacost_encoding{}".format(level)):
            if level > 0:
                d_lam = avg_pool3d(d_encoded[level - 1], 2)

            with tf.variable_scope("weights{}".format(level)):
                w1_d = conv_weight_variable(
                    "w1_d", [5, 5, 5, nclasses, nclasses], stddev=0.01*lam)
                w2_d = conv_weight_variable(
                    "w2_d", [5, 5, 5, nclasses, nclasses], stddev=0.01*lam)
                w3_d = conv_weight_variable(
                    "w3_d", [5, 5, 5, nclasses, nclasses], stddev=0.01*lam)
                b1_d = bias_weight_variable("b1_d", [nclasses])
                b2_d = bias_weight_variable("b2_d", [nclasses])
                b3_d = bias_weight_variable("b3_d", [nclasses])

            d_residual = conv3d(d_lam, w1_d)
            d_residual = tf.nn.relu(d_residual + b1_d)
            d_residual = conv3d(d_residual, w2_d)
            d_residual = tf.nn.relu(d_residual + b2_d)
            d_residual = conv3d(d_residual, w3_d,
                                name="d_encoded{}".format(level))
            d_residual += b3_d

            d_encoded.append(d_lam + d_residual)

    # Create a copy of the placeholders for the loop variables.
    u_loop = list(u)
    u_loop_= list(u_)
    m_loop = list(m)
    l_loop = list(l)

    for iter in range(niter):
        u_loop, u_loop_, m_loop, l_loop = pd.primal_dual(
            u_loop, u_loop_, m_loop, l_loop, d_encoded, sig, tau, iter)

    probs = u_loop

    for level in range(nlevels):
        u_loop[level] = tf.identity(
            u_loop[level], name="u_final{}".format(level))
        u_loop_[level] = tf.identity(
            u_loop_[level], name="u_final_{}".format(level))
        m_loop[level] = tf.identity(
            m_loop[level], name="m_final{}".format(level))
        l_loop[level] = tf.identity(
            l_loop[level], name="l_final{}".format(level))

    for level in range(nlevels):
        with tf.name_scope("prob_decoding{}".format(level)):
            with tf.variable_scope("weights{}".format(level)):
                w1_p = conv_weight_variable(
                    "w1_p", [5, 5, 5, nclasses, nclasses],
                    stddev=0.01*softmax_scale)
                w2_p = conv_weight_variable(
                    "w2_p", [5, 5, 5, nclasses, nclasses],
                    stddev=0.01*softmax_scale)
                w3_p = conv_weight_variable(
                    "w3_p", [5, 5, 5, nclasses, nclasses],
                    stddev=0.01*softmax_scale)
                b1_p = bias_weight_variable("b1_p", [nclasses])
                b2_p = bias_weight_variable("b2_p", [nclasses])
                b3_p = bias_weight_variable("b3_p", [nclasses])

            probs_residual = conv3d(probs[level], w1_p)
            probs_residual = tf.nn.relu(probs_residual + b1_p)
            probs_residual = conv3d(probs_residual, w2_p)
            probs_residual = tf.nn.relu(probs_residual + b2_p)
            probs_residual = conv3d(probs_residual, w3_p)
            probs_residual += b3_p

            probs[level] = tf.nn.softmax(softmax_scale * probs[level]
                                         + probs_residual,
                                         name="probs{}".format(level))

    return probs, d, u, u_, m, l


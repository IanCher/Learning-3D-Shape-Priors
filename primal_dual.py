import tensorflow as tf 
import numpy as np 
from tf_ops import *
from multiscale import resize_volumes 

def update_lagrangian(u_, l, sig, level):
    assert len(u_) == len(l)

    with tf.name_scope("lagrange_update"):
        sum_u = tf.reduce_sum(u_[level], axis=4, keepdims=False)
        l[level] += sig * (sum_u - 1.0)


def update_dual(u_, m, sig, level):
    assert len(u_) == len(m)

    with tf.name_scope("dual_update"):
        with tf.variable_scope("weights{}".format(level), reuse=True):
            w1 = tf.get_variable("w1")
            w2 = tf.get_variable("w2")

        _, nrows, ncols, nslices, nclasses = \
            u_[level].get_shape().as_list()
        batch_size = tf.shape(u_[level])[0]

        if level + 1 < len(u_):
            grad_u1 = conv3d(u_[level], w1)
            grad_u2 = conv3d(u_[level + 1], w2)
            grad_u = grad_u1 + resize_volumes(grad_u2, 2, 2, 2)
        else:
            grad_u = conv3d(u_[level], w1)

        m[level] += sig * grad_u

        m_rshp = tf.reshape(m[level], [batch_size, nrows, ncols,
                                       nslices, nclasses, 3])

        m_norm = tf.norm(m_rshp, ord="euclidean", axis=5, keepdims=True)
        m_norm = tf.maximum(m_norm, 1.0)
        m_normalized = tf.divide(m_rshp, m_norm)

        m[level] = tf.reshape(m_normalized, [batch_size, nrows, ncols,
                                             nslices, nclasses * 3])


def update_primal(u, m, l, d, tau, level):
    assert len(u) == len(m)
    assert len(u) == len(l)
    assert len(u) == len(d)

    with tf.name_scope("primal_update"):
        with tf.variable_scope("weights{}".format(level), reuse=True):
            w1 = tf.get_variable("w1")
            w2 = tf.get_variable("w2")

        _, nrows, ncols, nslices, nclasses = u[level].get_shape().as_list()
        batch_size = tf.shape(u[level])[0]

        if level + 1 < len(u):
            div_m1 = conv3d_adj(m[level], w1, nclasses)
            div_m2 = conv3d_adj(m[level + 1], w2, nclasses)
            div_m = div_m1 + resize_volumes(div_m2, 2, 2, 2)
        else:
            div_m = conv3d_adj(m[level], w1, nclasses)

        l_rshp = tf.reshape(l[level], [batch_size, nrows, ncols, nslices, 1])

        u[level] -= tau * (d[level] + l_rshp + div_m)

        u[level] = tf.minimum(1.0, tf.maximum(u[level], 0.0))


def primal_dual(u, u_, m, l, d, sig, tau, iter):
    u_0 = list(u)

    for level in list(range(len(u)))[::-1]:
        with tf.name_scope("primal_dual_iter{}_level{}".format(iter, level)):
            update_dual(u_, m, sig, level)

            update_lagrangian(u_, l, sig, level)

            update_primal(u, m, l, d, tau, level)

            u_[level] = 2 * u[level] - u_0[level]

    return u, u_, m, l


def initialize_pd_vars(nlevels, nrows, ncols, nslices, batch_size, nclasses):
    u_init  = []
    u_init_ = []
    m_init  = []
    l_init  = []

    for level in range(nlevels):
        factor = 2 ** level

        assert nrows % factor == 0
        assert ncols % factor == 0
        assert nslices % factor == 0

        nrows_level = nrows // factor
        ncols_level = ncols // factor
        nslices_level = nslices // factor

        u_init.append(np.empty(
            [batch_size, nrows_level, ncols_level, nslices_level, nclasses],
            dtype=np.float32)
        )
        u_init_.append(np.empty(
            [batch_size, nrows_level, ncols_level, nslices_level, nclasses], 
            dtype=np.float32)
        )
        m_init.append(np.empty(
            [batch_size, nrows_level, ncols_level, nslices_level, 3 * nclasses],
            dtype=np.float32)
        )
        l_init.append(np.empty(
            [batch_size, nrows_level, ncols_level, nslices_level],
            dtype=np.float32)
        )
    return u_init, u_init_, m_init, l_init

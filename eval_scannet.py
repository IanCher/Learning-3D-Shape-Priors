import os
import glob
import numpy as np
import tensorflow as tf

from model import build_model
from utils import mkdir_if_not_exists, extract_mesh_marching_cubes
from tf_loss_ops import *
from data_gen import build_data_generator
from summaries import *
from primal_dual import initialize_pd_vars
from parse_args import parse_args_eval
from train_scannet_final import create_feed_dict


def eval_model(checkpoint_path, datacost_path, params):
    niter_steps = params["niter_steps"]
    nlevels = params["nlevels"]

    datacost = np.load(datacost_path)
    resolution = datacost["resolution"]
    datacost = datacost["volume"]

    orig_shape = datacost.shape
    datacost = datacost[:datacost.shape[0]-(datacost.shape[0]%(2**(nlevels-1))),
                        :datacost.shape[1]-(datacost.shape[1]%(2**(nlevels-1))),
                        :datacost.shape[2]-(datacost.shape[2]%(2**(nlevels-1)))]

    print("Cropping datacost from", orig_shape, "to", datacost.shape)

    nrows = datacost.shape[0]
    ncols = datacost.shape[1]
    nslices = datacost.shape[2]
    nclasses = datacost.shape[3]

    params["nrows"] = nrows
    params["ncols"] = ncols
    params["nslices"] = nslices

    print("Building model")
    build_model(params)

    with tf.Session() as sess:
        print("Reading checkpoint")
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint(checkpoint_path))

        print("Initializing variables")

        graph = tf.get_default_graph()

        u_init = []
        u_init_ = []
        m_init = []
        l_init = []
        for level in range(nlevels):
            factor = 2 ** level
            assert nrows % factor == 0
            assert ncols % factor == 0
            assert nslices % factor == 0
            nrows_level = nrows // factor
            ncols_level = ncols // factor
            nslices_level = nslices // factor
            u_init.append(np.full([1, nrows_level, ncols_level,
                                   nslices_level, nclasses], 1.0 / nclasses,
                                   dtype=np.float32))
            u_init_.append(np.full([1, nrows_level, ncols_level,
                                    nslices_level, nclasses], 1.0 / nclasses,
                                    dtype=np.float32))
            m_init.append(np.zeros([1, nrows_level, ncols_level,
                                    nslices_level, 3 * nclasses],
                                   dtype=np.float32))
            l_init.append(np.zeros([1, nrows_level, ncols_level,
                                    nslices_level],
                                   dtype=np.float32))

        d = graph.get_tensor_by_name("d:0")
        p = graph.get_tensor_by_name("prob_decoding0/probs0:0")

        u = []
        u_ = []
        m = []
        l = []
        u_final = []
        u_final_ = []
        m_final = []
        l_final = []
        for level in range(nlevels):
            u.append(graph.get_tensor_by_name("u{}:0".format(level)))
            u_.append(graph.get_tensor_by_name("u_{}:0".format(level)))
            m.append(graph.get_tensor_by_name("m{}:0".format(level)))
            l.append(graph.get_tensor_by_name("l{}:0".format(level)))
            u_final.append(graph.get_tensor_by_name("u_final{}:0".format(level)))
            u_final_.append(graph.get_tensor_by_name("u_final_{}:0".format(level)))
            m_final.append(graph.get_tensor_by_name("m_final{}:0".format(level)))
            l_final.append(graph.get_tensor_by_name("l_final{}:0".format(level)))

        for step in range(niter_steps):
            print("  Step", step + 1, "/", niter_steps)
            feed_dict = {}
            feed_dict[d] = datacost[None]
            for level in range(nlevels):
                feed_dict[u[level]] = u_init[level]
                feed_dict[u_[level]] = u_init_[level]
                feed_dict[m[level]] = m_init[level]
                feed_dict[l[level]] = l_init[level]

            probs, u_init, u_init_, m_init, l_init = sess.run(
                [p, u_final, u_final_, m_final, l_final], feed_dict=feed_dict)

    return probs[0]


def main():
    args = parse_args_eval()

    np.random.seed(0)
    tf.set_random_seed(0)

    tf.logging.set_verbosity(tf.logging.INFO)

    params = {
        "nlevels": args.nlevels,
        "nclasses": args.nclasses,
        "nrows": args.nrows,
        "ncols": args.ncols,
        "nslices": args.nslices,
        "niter": args.niter,
        "niter_steps": args.niter_steps,
        "sig": args.sig,
        "tau": args.tau,
        "lam": args.lam,
        "softmax_scale": args.softmax_scale,
    }

    probs = eval_model(args.checkpoint_path, args.datacost_path, params)

    mkdir_if_not_exists(args.output_path)

    np.savez_compressed(
        os.path.join(args.output_path, "probs.npz"), probs=probs
    )

    if args.label_map_path:
        label_names = {}
        label_colors = {}
        with open(args.label_map_path, "r") as fid:
            for line in fid:
                line = line.strip()
                if not line:
                    continue
                label = int(line.split(":")[0].split()[0])
                name = line.split(":")[0].split()[1]
                color = tuple(map(int, line.split(":")[1].split()))
                label_names[label] = name
                label_colors[label] = color

    for label in range(probs.shape[-1]):
        if args.label_map_path:
            path = os.path.join(args.output_path,
                                "{}-{}.ply".format(label, label_names[label]))
            color = label_colors[label]
        else:
            path = os.path.join(args.output_path, "{}.ply".format(label))
            color = None

        extract_mesh_marching_cubes(path, probs[..., label], color=color)


if __name__ == "__main__":
    main()
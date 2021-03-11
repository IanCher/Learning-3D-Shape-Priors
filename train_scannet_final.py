import os
import glob
import numpy as np
import tensorflow as tf

from model import build_model
from utils import mkdir_if_not_exists
from tf_loss_ops import *
from data_gen import build_data_generator
from summaries import *
from primal_dual import initialize_pd_vars
from parse_args import parse_args


def create_feed_dict(
    train_data, datacost, groundtruth, nlevels, nclasses, 
    u_init, u_init_, m_init, l_init, u, u_, m, l):
    
    datacost_batch, groundtruth_batch = train_data

    num_batch_samples = datacost_batch.shape[0]

    feed_dict = {}

    feed_dict[datacost] = datacost_batch
    feed_dict[groundtruth] = groundtruth_batch

    for level in range(nlevels):
        u_init[level][:] = 1.0 / nclasses
        u_init_[level][:] = 1.0 / nclasses
        m_init[level][:] = 0.0
        l_init[level][:] = 0.0
        feed_dict[u[level]] = u_init[level][:num_batch_samples]
        feed_dict[u_[level]] = u_init_[level][:num_batch_samples]
        feed_dict[m[level]] = m_init[level][:num_batch_samples]
        feed_dict[l[level]] = l_init[level][:num_batch_samples]
    return feed_dict


def train_model(scene_path, scene_train_list_path, 
    scene_val_list_path, model_path, params):

    # Read parameters
    batch_size   = params["batch_size"]
    nlevels      = params["nlevels"]
    nrows        = params["nrows"]
    ncols        = params["ncols"]
    nslices      = params["nslices"]
    nclasses     = params["nclasses"]
    val_nbatches = params["val_nbatches"]

    # Create output paths
    log_path = os.path.join(model_path, "logs")
    checkpoint_path = os.path.join(model_path, "ckpts")

    mkdir_if_not_exists(log_path)
    mkdir_if_not_exists(checkpoint_path)

    # Data gemerators
    train_data_generator = build_data_generator(
        scene_path, scene_train_list_path, params
    )

    val_params = dict(params)
    val_params["epoch_npasses"] = -1
    val_data_generator = build_data_generator(
        scene_path, scene_val_list_path, val_params
    )

    # Build model
    probs, datacost, u, u_, m, l = build_model(params)
    groundtruth = tf.placeholder(tf.float32, probs[0].shape, name="groundtruth")

    # Initial pd variables
    u_init, u_init_, m_init, l_init = initialize_pd_vars(
        nlevels, nrows, ncols, nslices, batch_size, nclasses
    )

    # Loss operators
    loss_op = categorical_crossentropy(groundtruth, probs[0], params)
    freespace_accuracy_op, occupied_accuracy_op, semantic_accuracy_op = \
        classification_accuracy(groundtruth, probs[0])

    # Train operators
    optimizer = tf.train.AdamOptimizer(learning_rate=params["learning_rate"])
    train_op = optimizer.minimize(loss_op)

    # Create summaries
    summary_op, summaries_dict = create_summaries()

    # Create model saver
    model_saver = tf.train.Saver(save_relative_paths=True)
    train_saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=2,
                                 save_relative_paths=True, pad_step_number=True)

    with tf.Session() as sess:
        log_writer = tf.summary.FileWriter(log_path)
        log_writer.add_graph(sess.graph)

        sess.run(tf.global_variables_initializer())

        model_saver.save(sess, os.path.join(checkpoint_path, "initial"),
                         write_meta_graph=True)

        for epoch in range(params["nepochs"]):

            train_loss_values = []
            train_freespace_accuracy_values = []
            train_occupied_accuracy_values  = []
            train_semantic_accuracy_values  = []

            batch = 0
            while True:
                train_data = next(train_data_generator)

                # Check if epoch finished.
                if train_data is None:
                    break

                # Create feed dictionnary
                feed_dict = create_feed_dict(
                    train_data, datacost, groundtruth, nlevels, nclasses, 
                    u_init, u_init_, m_init, l_init, u, u_, m, l
                )

                # Run the graph
                (
                    _,
                    loss,
                    freespace_accuracy,
                    occupied_accuracy,
                    semantic_accuracy
                ) = sess.run(
                    [
                        train_op,
                        loss_op,
                        freespace_accuracy_op,
                        occupied_accuracy_op,
                        semantic_accuracy_op
                    ], 
                    feed_dict=feed_dict
                )

                # Gather accuracies
                train_loss_values.append(loss)
                train_freespace_accuracy_values.append(freespace_accuracy)
                train_occupied_accuracy_values.append(occupied_accuracy)
                train_semantic_accuracy_values.append(semantic_accuracy)

                # Print intermediate results
                print_train_results(
                    epoch, batch, loss, freespace_accuracy, 
                    occupied_accuracy, semantic_accuracy
                )

                batch += 1

            # Compute mean accuracies
            train_loss_value = np.nanmean(train_loss_values)
            train_freespace_accuracy_value = np.nanmean(train_freespace_accuracy_values)
            train_occupied_accuracy_value  = np.nanmean(train_occupied_accuracy_values)
            train_semantic_accuracy_value  = np.nanmean(train_semantic_accuracy_values)

            # VALIDATION
            val_loss_values = []
            val_freespace_accuracy_values = []
            val_occupied_accuracy_values = []
            val_semantic_accuracy_values = []

            for _ in range(val_nbatches):
                val_data = next(val_data_generator)

                # Create feed dictionnary
                feed_dict = create_feed_dict(
                    val_data, datacost, groundtruth, nlevels, nclasses, 
                    u_init, u_init_, m_init, l_init, u, u_, m, l
                )

                # Run the graph
                (
                    loss,
                    freespace_accuracy,
                    occupied_accuracy,
                    semantic_accuracy
                ) = sess.run(
                    [
                        loss_op,
                        freespace_accuracy_op,
                        occupied_accuracy_op,
                        semantic_accuracy_op
                    ],
                    feed_dict=feed_dict
                )

                # Gather accuracies
                val_loss_values.append(loss)
                val_freespace_accuracy_values.append(freespace_accuracy)
                val_occupied_accuracy_values.append(occupied_accuracy)
                val_semantic_accuracy_values.append(semantic_accuracy)

            val_loss_value = np.nanmean(val_loss_values)
            val_freespace_accuracy_value = np.nanmean(val_freespace_accuracy_values)
            val_occupied_accuracy_value  = np.nanmean(val_occupied_accuracy_values)
            val_semantic_accuracy_value  = np.nanmean(val_semantic_accuracy_values)

            # Print results
            print_val_results(
                val_loss_value, val_freespace_accuracy_value, 
                val_occupied_accuracy_value, val_semantic_accuracy_value
            )

            # Save summary
            summary_val ={
                "train_loss":train_loss_value,
                "train_freespace_accuracy":train_freespace_accuracy_value,
                "train_occupied_accuracy" :train_occupied_accuracy_value,
                "train_semantic_accuracy" :train_semantic_accuracy_value,
                "val_loss":val_loss_value,
                "val_freespace_accuracy":val_freespace_accuracy_value,
                "val_occupied_accuracy" :val_occupied_accuracy_value,
                "val_semantic_accuracy" :val_semantic_accuracy_value,
            }

            save_summary(
                sess, summary_op, summaries_dict, 
                summary_val, log_writer, epoch
            )

            # Save model / checkpoints
            train_saver.save(sess, os.path.join(checkpoint_path, "checkpoint"),
                             global_step=epoch, write_meta_graph=False)

        model_saver.save(sess, os.path.join(checkpoint_path, "final"),
                         write_meta_graph=True)


def main():
    args = parse_args()

    np.random.seed(0)
    tf.set_random_seed(0)

    tf.logging.set_verbosity(tf.logging.INFO)

    params = {
        "nepochs": args.nepochs,
        "epoch_npasses": args.epoch_npasses,
        "val_nbatches": args.val_nbatches,
        "batch_size": args.batch_size,
        "nlevels": args.nlevels,
        "nclasses": args.nclasses,
        "nrows": args.nrows,
        "ncols": args.ncols,
        "nslices": args.nslices,
        "niter": args.niter,
        "sig": args.sig,
        "tau": args.tau,
        "lam": args.lam,
        "learning_rate": args.learning_rate,
        "softmax_scale": args.softmax_scale,
    }

    train_model(args.scene_path, args.scene_train_list_path,
                args.scene_val_list_path, args.model_path, params)


if __name__ == "__main__":
    main()

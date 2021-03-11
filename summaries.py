import tensorflow as tf 


def create_summaries():

    train_loss_summary = \
        tf.placeholder(tf.float32, name="train_loss_summary")
    train_freespace_accuracy_summary = \
        tf.placeholder(tf.float32, name="train_freespace_accuracy_summary")
    train_occupied_accuracy_summary = \
        tf.placeholder(tf.float32, name="train_occupied_accuracy_summary")
    train_semantic_accuracy_summary = \
        tf.placeholder(tf.float32, name="train_semantic_accuracy_summary")

    tf.summary.scalar("train_loss", train_loss_summary)
    tf.summary.scalar("train_freespace_accuracy", train_freespace_accuracy_summary)
    tf.summary.scalar("train_occupied_accuracy" , train_occupied_accuracy_summary)
    tf.summary.scalar("train_semantic_accuracy" , train_semantic_accuracy_summary)

    val_loss_summary = \
        tf.placeholder(tf.float32, name="val_loss_summary")
    val_freespace_accuracy_summary = \
        tf.placeholder(tf.float32, name="val_freespace_accuracy_summary")
    val_occupied_accuracy_summary = \
        tf.placeholder(tf.float32, name="val_occupied_accuracy_summary")
    val_semantic_accuracy_summary = \
        tf.placeholder(tf.float32, name="val_semantic_accuracy_summary")

    tf.summary.scalar("val_loss", val_loss_summary)
    tf.summary.scalar("val_freespace_accuracy", val_freespace_accuracy_summary)
    tf.summary.scalar("val_occupied_accuracy" , val_occupied_accuracy_summary)
    tf.summary.scalar("val_semantic_accuracy" , val_semantic_accuracy_summary)

    summary_op = tf.summary.merge_all()

    # Combine all summaries in a dictionnary
    summaries_dict = {
        "train_loss"              : train_loss_summary,
        "train_freespace_accuracy": train_freespace_accuracy_summary,
        "train_occupied_accuracy" : train_occupied_accuracy_summary,
        "train_semantic_accuracy" : train_semantic_accuracy_summary,
        "val_loss"                : val_loss_summary,
        "val_freespace_accuracy"  : val_freespace_accuracy_summary,
        "val_occupied_accuracy"   : val_occupied_accuracy_summary,
        "val_semantic_accuracy"   : val_semantic_accuracy_summary,
    }

    return summary_op, summaries_dict


def save_summary(sess, summary_op, summaries_dict, summary_val, log_writer, epoch):
    summary = sess.run(
        summary_op,
        feed_dict={
            summaries_dict["train_loss"]:
                summary_val["train_loss"],
            summaries_dict["train_freespace_accuracy"]:
                summary_val["train_freespace_accuracy"],
            summaries_dict["train_occupied_accuracy"]:
                summary_val["train_occupied_accuracy"],
            summaries_dict["train_semantic_accuracy"]:
                summary_val["train_semantic_accuracy"],
            summaries_dict["val_loss"]:
                summary_val["val_loss"],
            summaries_dict["val_freespace_accuracy"]:
                summary_val["val_freespace_accuracy"],
            summaries_dict["val_occupied_accuracy"]:
                summary_val["val_occupied_accuracy"],
            summaries_dict["val_semantic_accuracy"]:
                summary_val["val_semantic_accuracy"],
        }
    )

    log_writer.add_summary(summary, epoch)


def print_val_results(
    val_loss_value, val_freespace_accuracy_value, 
    val_occupied_accuracy_value, val_semantic_accuracy_value
):
    print()
    print(78 * "#")
    print()

    print("Validation\n"
          "  Loss:                  {}\n"
          "  Free Space Accuracy:   {}\n"
          "  Occupied Accuracy:     {}\n"
          "  Semantic Accuracy:     {}".format(
          val_loss_value,
          val_freespace_accuracy_value,
          val_occupied_accuracy_value,
          val_semantic_accuracy_value))

    print()
    print(78 * "#")
    print(78 * "#")
    print()


def print_train_results(
    epoch, batch, loss, freespace_accuracy, 
    occupied_accuracy, semantic_accuracy
):
    print("Epoch: {}, "
          "Batch: {}\n"
          "  Loss:                  {}\n"
          "  Free Space Accuracy:   {}\n"
          "  Occupied Accuracy:     {}\n"
          "  Semantic Accuracy:     {}".format(
          epoch + 1,
          batch + 1,
          loss,
          freespace_accuracy,
          occupied_accuracy,
          semantic_accuracy)
    )
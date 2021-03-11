import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--scene_path", default="Scannet_Data")
    parser.add_argument("--scene_train_list_path", default="train.txt")
    parser.add_argument("--scene_val_list_path", default="test.txt")
    parser.add_argument("--model_path", default="Model")

    parser.add_argument("--nclasses", type=int, default=42)

    parser.add_argument("--nlevels", type=int, default=1)
    parser.add_argument("--nrows", type=int, default=24)
    parser.add_argument("--ncols", type=int, default=24)
    parser.add_argument("--nslices", type=int, default=24)

    parser.add_argument("--niter", type=int, default=60)
    parser.add_argument("--sig", type=float, default=0.2)
    parser.add_argument("--tau", type=float, default=0.2)
    parser.add_argument("--lam", type=float, default=1.0)

    parser.add_argument("--batch_size", type=int, default=3)
    parser.add_argument("--nepochs", type=int, default=1000)
    parser.add_argument("--epoch_npasses", type=int, default=1)
    parser.add_argument("--val_nbatches", type=int, default=15)
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--softmax_scale", type=float, default=10)

    return parser.parse_args()

def parse_args_eval():
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--datacost_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--label_map_path")

    parser.add_argument("--nlevels", type=int, default=1)
    parser.add_argument("--nclasses", type=int, default=42)
    parser.add_argument("--nrows", type=int, default=24)
    parser.add_argument("--ncols", type=int, default=24)
    parser.add_argument("--nslices", type=int, default=24)

    parser.add_argument("--niter", type=int, default=60)
    parser.add_argument("--sig", type=float, default=0.2)
    parser.add_argument("--tau", type=float, default=0.2)
    parser.add_argument("--lam", type=float, default=1.0)

    parser.add_argument("--batch_size", type=int, default=3)
    parser.add_argument("--softmax_scale", type=float, default=10)
    parser.add_argument("--niter_steps", type=int, default=1)

    return parser.parse_args()

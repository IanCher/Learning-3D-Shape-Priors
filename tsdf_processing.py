import argparse
import glob
import numpy as np
import os 
import shutil
from TSDF.tsdf_volume import TSDFVolume
from zipfile import ZipFile


useless_path_in_zip = "home/iancher/Remote/leonhard-scratch/Scannet_Data_Processed"


def read_num_labels(labels_file):
    num_labels = 0
    with open(labels_file, "r") as fid:
        for line in fid:
            if line.strip() and not "freespace" in line:
                num_labels += 1

    return num_labels


def write_ply(path, points, color):
    with open(path, "w") as fid:
        fid.write("ply\n")
        fid.write("format ascii 1.0\n")
        fid.write("element vertex {}\n".format(points.shape[0]))
        fid.write("property float x\n")
        fid.write("property float y\n")
        fid.write("property float z\n")
        fid.write("property uchar diffuse_red\n")
        fid.write("property uchar diffuse_green\n")
        fid.write("property uchar diffuse_blue\n")
        fid.write("end_header\n")
        for i in range(points.shape[0]):
            fid.write("{} {} {} {} {} {}\n".format(points[i, 0], points[i, 1],
                                                   points[i, 2], *color))


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", type=str, default="scenes")
    parser.add_argument("--files_list", type=str, default="Test_Box/train.txt")
    parser.add_argument("--labels_file", type=str, default="Test_Box/labels.txt")
    
    parser.add_argument("--resolution", type=float, default=0.05)
    parser.add_argument("--resolution_factor", type=int, default=2)
    parser.add_argument("--frame_rate", type=int, default=100)

    return parser.parse_args()


def main():
    args = parse_args()

    # Read number of labels labels
    num_labels = read_num_labels(args.labels_file)

    # Process each scenes
    with open(args.files_list, 'r') as fid:
        scene_name = fid.readline().splitlines()[0]
        scene_path = os.path.join(args.data_path, scene_name)
        
        # Open the bbox
        bbox = np.loadtxt(os.path.join(scene_path, "bbox.txt"))

        tsdf_volume = TSDFVolume(
            num_labels, bbox, args.resolution, args.resolution_factor
        )

        # Unzip images
        img_zip_file = os.path.join(scene_path, 'images.zip')
        temp_zip_folder = os.path.join(scene_path, 'images')

        with ZipFile(img_zip_file, 'r') as zipObj:
            images_path = os.path.join(
                temp_zip_folder, useless_path_in_zip, 
                "{}/images".format(scene_name)
            )
            zipObj.extractall(temp_zip_folder)

        image_paths = sorted(glob.glob(os.path.join(images_path, "*.npz")))

        for i, image_path in enumerate(image_paths):
            if i % args.frame_rate != 0:
                continue

            print("Processing {} [{}/{}]".format(
                os.path.basename(image_path), i + 1, len(image_paths)
                )
            )
            image = np.load(image_path)

            if image["label_map"].dtype == np.int32:
                label_map = np.zeros((image["label_map"].shape[0],
                                    image["label_map"].shape[1],
                                    num_labels), dtype=np.float32)
                rr, cc = np.mgrid[:label_map.shape[0], :label_map.shape[1]]
                label_map[(rr.ravel(), cc.ravel(), image["label_map"].ravel())] = 1
            else:
                label_map = image["label_map"]

            tsdf_volume.fuse(
                image["depth_proj_matrix"], image["label_proj_matrix"],
                image["depth_map"], label_map
            )

        occupied_volume_idxs = np.column_stack(
            np.where(np.sum(tsdf_volume.get_volume()[..., :-1], axis=-1) < 0)
        )
        write_ply(
            os.path.join(scene_path, "datacost.ply"),
            occupied_volume_idxs, color=[255, 0, 0]
        )

        np.savez(
            os.path.join(scene_path, "datacost.npz"),
            volume=tsdf_volume.get_volume(),
            resolution=args.resolution
        )

        shutil.rmtree(temp_zip_folder)





            

if __name__ == "__main__":
    main()
import os
import glob
import shutil
import argparse
import h5py
import numpy as np
import pandas as pd
import skimage.io
from pyntcloud import PyntCloud
from pyntcloud.samplers.base import Sampler
from pyntcloud.geometry.areas import triangle_area_multi

from tsdf_volume import TSDFVolume


# NYU40_LABELS = [
#     "unknown",
#     "wall",
#     "floor",
#     "cabinet",
#     "bed",
#     "chair",
#     "sofa",
#     "table",
#     "door",
#     "window",
#     "bookshelf",
#     "picture",
#     "counter",
#     "desk",
#     "curtain",
#     "refridgerator",
#     "shower curtain",
#     "toilet",
#     "sink",
#     "bathtub",
#     "otherfurniture",
# ]

NYU40_LABEL_COLORS = (
    ("wall", (176, 135, 93)), # #B0875D
    ("floor", (175, 183, 255)), # #AFB7FF
    ("cabinet", (205, 255, 10)), # #CDFF0A
    ("bed", (138, 0, 123)), # #8A007B
    ("chair", (139, 255, 65)), # #8BFF41
    ("sofa", (98, 60, 1)), # #623C01
    ("table", (101, 51, 111)), # #65336F
    ("door", (20, 161, 106)), # #14A16A
    ("window", (13, 104, 166)), # #0D68A6
    ("bookshelf", (15, 125, 138)), # #0F7D8A
    ("picture", (88, 84, 110)), # #58546E
    ("counter", (254, 226, 10)), # #FEE20A
    ("blinds", (97, 44, 162)), # #612CA2
    ("desk", (33, 255, 253)), # #21FFFD
    ("shelves", (0, 14, 52)), # #000E34
    ("curtain", (253, 142, 255)), # #FD8EFF
    ("dresser", (179, 119, 141)), # #B3778D
    ("pillow", (78, 8, 0)), # #4E0800
    ("mirror", (6, 55, 66)), # #063742
    ("floor mat", (254, 213, 83)), # #FED553
    ("clothes", (9, 90, 254)), # #095AFE
    ("ceiling", (51, 0, 33)), # #330021
    ("books", (25, 198, 255)), # #19C6FF
    ("refridgerator", (224, 64, 176)), # #E040B0
    ("television", (128, 199, 191)), # #80C7BF
    ("paper", (253, 126, 108)), # #FD7E6C
    ("towel", (80, 160, 61)), # #50A03D
    ("shower curtain", (11, 84, 2)), # #0B5402
    ("box", (255, 234, 226)), # #FFEAE2
    ("whiteboard", (16, 133, 255)), # #1085FF
    ("person", (252, 0, 69)), # #FC0045
    ("night stand", (176, 204, 128)), # #B0CC80
    ("toilet", (115, 157, 4)), # #739D04
    ("sink", (129, 0, 44)), # #81002C
    ("lamp", (251, 0, 140)), # #FB008C
    ("bathtub", (172, 118, 5)), # #AC7605
    ("bag", (23, 176, 19)), # #17B013
    ("otherstructure", (0, 0, 83)), # #000053
    ("otherfurniture", (0, 0, 255)), # #0000FF
    ("otherprop", (251, 0, 244)), # #FB00F4
    ("unknown", (0, 0, 0)), # #000000
)

NYU40_LABELS = tuple(label for label, _ in NYU40_LABEL_COLORS)
NYU40_COLORS = tuple(color for _, color in NYU40_LABEL_COLORS)


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


def mkdir_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--label_map_path", required=True)
    parser.add_argument("--overwrite", type=int, default=False)
    parser.add_argument("--resolution", type=float, default=0.05)
    return parser.parse_args()


def main():
    args = parse_args()

    mkdir_if_not_exists(args.output_path)
    mkdir_if_not_exists(os.path.join(args.output_path, "images"))

    # Load the label mapping.
    label_mapping = {}
    with open(args.label_map_path, "r") as fid:
        for i, line in enumerate(fid):
            elems = list(map(lambda x: x.strip(), line.split("\t")))
            if i == 0:
                assert elems[7] == "nyu40class"
            else:
                nyu40class = elems[7]
                if nyu40class in NYU40_LABELS:
                    label = int(elems[0])
                    label_mapping[label] = NYU40_LABELS.index(nyu40class)
    assert 0 not in label_mapping
    label_mapping[0] = NYU40_LABELS.index("unknown")

    # Load the extrinsic calibration between color and depth sensor.
    info_path = glob.glob(os.path.join(args.scene_path, "*.txt"))
    color_to_depth_T = None
    assert len(info_path) == 1
    with open(info_path[0], "r") as fid:
        for line in fid:
            if line.startswith("colorToDepthExtrinsics"):
                color_to_depth_T = \
                    np.array(list(map(float, line.split("=")[1].split())))
                color_to_depth_T = color_to_depth_T.reshape(4, 4)
    if color_to_depth_T is None:
        color_to_depth_T = np.eye(4)

    # Load the intrinsic calibration parameters.
    with open(os.path.join(args.scene_path, "sensor/_info.txt"), "r") as fid:
        for line in fid:
            if line.startswith("m_calibrationColorIntrinsic"):
                label_K = np.array(list(map(float, line.split("=")[1].split())))
                label_K = label_K.reshape(4, 4)[:3, :3]
            elif line.startswith("m_calibrationDepthIntrinsic"):
                depth_K = np.array(list(map(float, line.split("=")[1].split())))
                depth_K = depth_K.reshape(4, 4)[:3, :3]
            elif line.startswith("m_calibrationColorExtrinsic") or \
                    line.startswith("m_calibrationDepthExtrinsic"):
                assert line.split("=")[1].strip() \
                       == "1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1"

    # Load the semantic groundtruth mesh and convert it to a point cloud.
    groundtruth = PyntCloud.from_file(
        glob.glob(os.path.join(args.scene_path, "*_vh_clean_2.labels.ply"))[0])
    # groundtruth = RandomMesh(groundtruth, n=1000000, labels=True).compute()

    # Compute the bounding box of the ground truth.
    minx = groundtruth.points.x.min()
    miny = groundtruth.points.y.min()
    minz = groundtruth.points.z.min()
    maxx = groundtruth.points.x.max()
    maxy = groundtruth.points.y.max()
    maxz = groundtruth.points.z.max()

    # Extend the bounding box by a stretching factor.
    diffx = maxx - minx
    diffy = maxy - miny
    diffz = maxz - minz
    minx -= 0.05 * diffx
    maxx += 0.05 * diffx
    miny -= 0.05 * diffy
    maxy += 0.05 * diffy
    minz -= 0.05 * diffz
    maxz += 0.05 * diffz

    # Write the bounding box.
    bbox = np.array(((minx, maxx), (miny, maxy), (minz, maxz)),
                    dtype=np.float32)
    np.savetxt(os.path.join(args.output_path, "bbox.txt"), bbox)

    tsdf_volume = TSDFVolume(num_labels=len(NYU40_LABELS), bbox=bbox,
                             resolution=args.resolution, resolution_factor=2)

    # Process the frames in the scene.
    pose_paths = sorted(glob.glob(
        os.path.join(args.scene_path, "sensor/*.pose.txt")))
    for i, pose_path in enumerate(pose_paths):
        image_id = int(os.path.basename(pose_path)[6:-9])

        print("Processing frame {} [{}/{}]".format(
            image_id, i + 1, len(pose_paths)))

        output_path = os.path.join(args.output_path,
                                   "images/{:06d}.npz".format(image_id))

        if args.overwrite or not os.path.exists(output_path):
            depth_map_path = os.path.join(
                args.scene_path,
                "sensor/frame-{:06d}.depth.pgm".format(image_id))
            label_map_path = os.path.join(
                args.scene_path,
                "label/{}.png".format(image_id))

            assert os.path.exists(depth_map_path)
            assert os.path.exists(label_map_path)

            pose = np.loadtxt(pose_path)

            proj_matrix = np.linalg.inv(pose)
            depth_proj_matrix = \
                np.dot(depth_K, np.dot(color_to_depth_T, proj_matrix)[:3])
            label_proj_matrix = np.dot(label_K, proj_matrix[:3])

            depth_proj_matrix = depth_proj_matrix.astype(np.float32)
            label_proj_matrix = label_proj_matrix.astype(np.float32)

            depth_map = skimage.io.imread(depth_map_path)
            depth_map = depth_map.astype(np.float32) / 1000

            label_map = skimage.io.imread(label_map_path)
            label_map_converted = np.full(label_map.shape, -1, dtype=np.int32)
            for label in np.unique(label_map):
                if label in label_mapping:
                    label_map_converted[label_map==label] = label_mapping[label]
            label_map = label_map_converted

            # Write the output into one combined NumPy file.
            np.savez_compressed(
                output_path,
                depth_proj_matrix=depth_proj_matrix,
                label_proj_matrix=label_proj_matrix,
                depth_map=depth_map,
                label_map=label_map)

        else:
            image = np.load(output_path)
            depth_proj_matrix = image["depth_proj_matrix"]
            label_proj_matrix = image["label_proj_matrix"]
            depth_map = image["depth_map"]
            label_map = image["label_map"]

        # Convert label map to per class probability map.
        label_probs = np.zeros((label_map.shape[0],
                                label_map.shape[1],
                                len(NYU40_LABELS)), dtype=np.float32)
        rr, cc = np.mgrid[:label_map.shape[0], :label_map.shape[1]]
        label_probs[(rr.ravel(), cc.ravel(), label_map.ravel())] = 1

        # Fuse the image in the TSDF volume.
        tsdf_volume.fuse(depth_proj_matrix, label_proj_matrix,
                         depth_map, label_probs)

    # Save the label and color mapping.
    with open(os.path.join(args.output_path, "labels.txt"), "w") as fid:
        for i, (label, color) in enumerate(NYU40_LABEL_COLORS):
            fid.write("{} {} : {} {} {}\n".format(i, label, *color))
        fid.write("{} freespace : 255 0 0\n".format(i + 1))

    # Save the groundtruth volume as a sparse matrix.
    tsdf_volume = tsdf_volume.get_volume()
    tsdf_volume_idxs = np.where(tsdf_volume != 0)
    tsdf_volume_values = tsdf_volume[tsdf_volume_idxs]
    tsdf_volume_idxs = np.row_stack(tsdf_volume_idxs)
    np.savez_compressed(os.path.join(args.output_path,
                                     "groundtruth_datacost.npz"),
                        resolution=args.resolution,
                        shape=tsdf_volume.shape,
                        idxs=tsdf_volume_idxs,
                        values=tsdf_volume_values)

    write_ply(os.path.join(args.output_path,
                           "groundtruth_datacost_occupied.ply"),
              np.column_stack(np.where(tsdf_volume[:, :, :, 0] < 0)) \
              * args.resolution, color=[255, 0, 0])
    write_ply(os.path.join(args.output_path,
                           "groundtruth_datacost_freespace.ply"),
              np.column_stack(np.where(tsdf_volume[:, :, :, 1] < 0)) \
              * args.resolution, color=[0, 255, 0])
    write_ply(os.path.join(args.output_path,
                           "groundtruth_datacost_unobserved.ply"),
              np.column_stack(np.where(np.sum(tsdf_volume, axis=-1) == 0)) \
              * args.resolution, color=[0, 0, 255])


if __name__ == "__main__":
    main()

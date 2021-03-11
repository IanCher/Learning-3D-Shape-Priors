import os
import sys
import struct
import shutil
import argparse
import collections
import numpy as np
import skimage.io
import skimage.transform
import nibabel.quaternions


CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"])
Camera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"])
Image = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
Point3D = collections.namedtuple(
    "Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])


CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=4, model_name="RADIAL", num_params=5),
    CameraModel(model_id=5, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=6, model_name="OPENCV", num_params=8),
    CameraModel(model_id=7, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=8, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=9, model_name="FOV", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12)
}
CAMERA_MODEL_IDS = dict([(camera_model.model_id, camera_model) \
                         for camera_model in CAMERA_MODELS])


def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file.
    :param fid:
    :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    :param endian_character: Any of {@, =, <, >, !}
    :return: Tuple of read and unpacked values.
    """
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


def read_cameras_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasBinary(const std::string& path)
        void Reconstruction::ReadCamerasBinary(const std::string& path)
    """
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for camera_line_index in range(num_cameras):
            camera_properties = read_next_bytes(
                fid, num_bytes=24, format_char_sequence="iiQQ")
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            model_name = CAMERA_MODEL_IDS[camera_properties[1]].model_name
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = CAMERA_MODEL_IDS[model_id].num_params
            params = read_next_bytes(fid, num_bytes=8*num_params,
                                     format_char_sequence="d"*num_params)
            cameras[camera_id] = Camera(id=camera_id,
                                        model=model_name,
                                        width=width,
                                        height=height,
                                        params=np.array(params))
        assert len(cameras) == num_cameras
    return cameras


def read_images_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesBinary(const std::string& path)
        void Reconstruction::WriteImagesBinary(const std::string& path)
    """
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for image_index in range(num_reg_images):
            binary_image_properties = read_next_bytes(
                fid, num_bytes=64, format_char_sequence="idddddddi")
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":   # look for the ASCII 0 entry
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            num_points2D = read_next_bytes(fid, num_bytes=8,
                                           format_char_sequence="Q")[0]
            x_y_id_s = read_next_bytes(fid, num_bytes=24*num_points2D,
                                       format_char_sequence="ddq"*num_points2D)
            xys = np.column_stack([tuple(map(float, x_y_id_s[0::3])),
                                   tuple(map(float, x_y_id_s[1::3]))])
            point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
            images[image_id] = Image(
                id=image_id, qvec=qvec, tvec=tvec,
                camera_id=camera_id, name=image_name,
                xys=xys, point3D_ids=point3D_ids)
    return images



def read_points3d_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DBinary(const std::string& path)
        void Reconstruction::WritePoints3DBinary(const std::string& path)
    """
    points3D = {}
    with open(path_to_model_file, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]
        for point_line_index in range(num_points):
            binary_point_line_properties = read_next_bytes(
                fid, num_bytes=43, format_char_sequence="QdddBBBd")

            point3D_id = binary_point_line_properties[0]
            xyz = np.array(binary_point_line_properties[1:4])
            rgb = np.array(binary_point_line_properties[4:7])
            error = np.array(binary_point_line_properties[7])

            track_length = read_next_bytes(
                fid, num_bytes=8, format_char_sequence="Q")[0]
            track_elems = read_next_bytes(
                fid, num_bytes=8*track_length,
                format_char_sequence="ii"*track_length)

            image_ids = np.array(tuple(map(int, track_elems[0::2])))
            point2D_idxs = np.array(tuple(map(int, track_elems[1::2])))

            points3D[point3D_id] = Point3D(
                id=point3D_id, xyz=xyz, rgb=rgb,
                error=error, image_ids=image_ids,
                point2D_idxs=point2D_idxs)
    return points3D


def read_depth_map(path):
    with open(path, "rb") as fid:
        width, height, channels = np.genfromtxt(fid, delimiter="&", max_rows=1,
                                                usecols=(0, 1, 2), dtype=int)
        fid.seek(0)
        num_delimiter = 0
        byte = fid.read(1)
        while True:
            if byte == b"&":
                num_delimiter += 1
                if num_delimiter >= 3:
                    break
            byte = fid.read(1)
        array = np.fromfile(fid, np.float32)
    array = array.reshape((width, height, channels), order="F")
    return np.transpose(array, (1, 0, 2)).squeeze()


def mkdir_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--label_color_map_path", required=True)
    parser.add_argument("--label_ext", default=".png")
    parser.add_argument("--depth_type", default="photometric")
    return parser.parse_args()


def main():
    args = parse_args()

    mkdir_if_not_exists(args.output_path)
    mkdir_if_not_exists(os.path.join(args.output_path, "images"))

    # Load the mapping from colors to labels.
    label_color_map = {}
    with open(args.label_color_map_path, "r") as fid:
        for line in fid:
            line = line.strip()
            if not line:
                continue
            label, color = line.split(":")
            color = color.strip().split(" ")
            assert len(color) == 1 or len(color) == 3
            label_color_map[int(label)] = tuple(map(int, color))

    # The number of labels is determined by the maximum label index.
    num_labels = max(label_color_map.keys()) + 1

    # Read the COLMAP sparse reconstruction.
    cameras = read_cameras_binary(os.path.join(args.input_path,
                                               "sparse/cameras.bin"))
    images = read_images_binary(os.path.join(args.input_path,
                                             "sparse/images.bin"))
    points3D = read_points3d_binary(os.path.join(args.input_path,
                                                 "sparse/points3D.bin"))

    # Compute robust bounding box for model from sparse points.
    xyz = np.row_stack([point3D.xyz for point3D in points3D.values()])
    minx, maxx = np.percentile(xyz[:, 0], [5, 95])
    miny, maxy = np.percentile(xyz[:, 1], [5, 95])
    minz, maxz = np.percentile(xyz[:, 2], [5, 95])
    diffx = maxx - minx
    diffy = maxy - miny
    diffz = maxz - minz
    minx -= 0.25 * diffx
    maxx += 0.25 * diffx
    miny -= 0.25 * diffy
    maxy += 0.25 * diffy
    minz -= 0.25 * diffz
    maxz += 0.25 * diffz

    # Write the bounding box to a text file.
    with open(os.path.join(args.output_path, "bbox.txt"), "w") as fid:
        fid.write("{} {}\n".format(minx, maxx))
        fid.write("{} {}\n".format(miny, maxy))
        fid.write("{} {}\n".format(minz, maxz))

    # Convert each reconstructed image.
    for i, image in enumerate(images.values()):
        print("Processing {} [{}/{}]".format(
              image.name, i + 1, len(images)))

        camera = cameras[image.camera_id]

        # Compose the projection matrix.
        assert camera.model == "PINHOLE"
        K = np.array([[camera.params[0], 0, camera.params[2]],
                      [0, camera.params[1], camera.params[3]],
                      [0, 0, 1]], dtype=np.double)
        P = np.column_stack([nibabel.quaternions.quat2mat(image.qvec),
                             image.tvec.T])
        proj_matrix = np.dot(K, P).astype(np.float32)

        # Read the depth map.
        depth_map = read_depth_map(os.path.join(args.input_path,
            "stereo/depth_maps/{}.{}.bin".format(image.name, args.depth_type)))

        # Load label colors for image.
        label_colors = skimage.io.imread(os.path.join(
            args.input_path, "labels", image.name + args.label_ext))

        if depth_map.shape != label_colors.shape:
            print("  WARNING: shape of depth map does not match label map "
                  "automatically resizing label map to fit depth map")
            label_colors = skimage.transform.resize(label_colors,
                depth_map.shape, order=0, mode="reflect", preserve_range=True)
            label_colors = label_colors.astype(np.uint8)

        label_colors = np.atleast_3d(label_colors)

        # Convert label colors to unique indices.
        label_idxs = np.zeros((label_colors.shape[0], label_colors.shape[1]),
                              dtype=np.uint32)
        for d in range(label_colors.shape[2]):
            label_idxs += label_colors[:, :, d] * 256**d

        # Lookup the correct label probability for each label color.
        label_map = np.zeros((label_colors.shape[0], label_colors.shape[1],
                              num_labels), dtype=np.float32)
        for label, label_color in label_color_map.items():
            assert len(label_color) == label_colors.shape[2]
            label_color_idx = 0
            for d in range(label_colors.shape[2]):
                label_color_idx += label_color[d] * 256**d
            label_map_idxs = list(np.where(label_idxs == label_color_idx))
            label_map_idxs.append(label * np.ones(label_map_idxs[0].size,
                                                  dtype=int))
            label_map[label_map_idxs] = 1

        # Write the output into one combined NumPy file.
        np.savez(os.path.join(args.output_path, "images", image.name + ".npz"),
                 label_proj_matrix=proj_matrix,
                 depth_proj_matrix=proj_matrix,
                 label_map=label_map,
                 depth_map=depth_map)


if __name__ == "__main__":
    main()

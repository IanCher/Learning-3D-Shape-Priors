import tensorflow as tf
import os
import numpy as np

def build_data_generator(scene_path, scene_list_path, params):
    epoch_npasses = params["epoch_npasses"]
    batch_size = params["batch_size"]
    nrows = params["nrows"]
    ncols = params["ncols"]
    nslices = params["nslices"]
    nclasses = params["nclasses"]

    scene_list = []
    with open(scene_list_path, "r") as fid:
        for line in fid:
            line = line.strip()
            if line:
                scene_list.append(line)

    # Load the data for all scenes.
    datacosts = []
    groundtruths = []
    for i, scene_name in enumerate(scene_list):
        print("Loading {} [{}/{}]".format(scene_name, i + 1, len(scene_list)))

        # if len(datacosts) == 5:
        #     break

        datacost_path = os.path.join(scene_path, scene_name, "datacost.npz")
        groundtruth_path = os.path.join(scene_path, scene_name, "groundtruth.npz")

        if not os.path.exists(datacost_path) or not os.path.exists(groundtruth_path):
            print("  Warning: datacost or groundtruth does not exist")
            print("  --> datacost_path: {}".format(datacost_path))
            print("  --> groundtruth_path: {}".format(groundtruth_path))
            continue

        datacost_data = np.load(datacost_path)
        datacost = datacost_data["volume"]

        groundtruth = np.load(groundtruth_path)
        groundtruth = groundtruth["probs"]

        # Make sure the data is compatible with the parameters.
        assert datacost.shape[3] == nclasses

        data_shape = np.minimum(datacost.shape, groundtruth.shape)
        datacost = datacost[:data_shape[0], :data_shape[1], :data_shape[2]]
        groundtruth = datacost[:data_shape[0], :data_shape[1], :data_shape[2]]
        
        try:
            assert datacost.shape == groundtruth.shape
        except:
            print("Data Shape: {}".format(datacost.shape))
            print("GT   Shape: {}".format(groundtruth.shape))
            return

        datacosts.append(datacost)
        groundtruths.append(groundtruth)

    idxs = np.arange(len(datacosts))

    batch_datacost = np.empty(
        (batch_size, nrows, ncols, nslices, nclasses), dtype=np.float32)
    batch_groundtruth = np.empty(
        (batch_size, nrows, ncols, nslices, nclasses), dtype=np.float32)

    npasses = 0

    while True:
        # Shuffle all data samples.
        np.random.shuffle(idxs)

        # One epoch iterates once over all scenes.
        for batch_start_idx in range(0, len(idxs), batch_size):
            # Determine the random scenes for current batch.
            batch_end_idx = min(batch_start_idx + batch_size, len(idxs))
            batch_idxs = idxs[batch_start_idx:batch_end_idx]

            # By default, set all voxels to unobserved.
            batch_datacost[:] = 0
            batch_groundtruth[:] = 1.0 / nclasses

            # Prepare data for random scenes in current batch.
            for i, idx in enumerate(batch_idxs):
                datacost = datacosts[idx]
                groundtruth = groundtruths[idx]

                # Determine a random crop of the input data.
                row_start = np.random.randint(
                    0, max(datacost.shape[0] - nrows, 0) + 1)
                col_start = np.random.randint(
                    0, max(datacost.shape[1] - ncols, 0) + 1)
                slice_start = np.random.randint(
                    0, max(datacost.shape[2] - nslices, 0) + 1)
                row_end = min(row_start + nrows, datacost.shape[0])
                col_end = min(col_start + ncols, datacost.shape[1])
                slice_end = min(slice_start + nslices, datacost.shape[2])

                # Copy the random crop of the data cost.
                batch_datacost[i,
                               :row_end-row_start,
                               :col_end-col_start,
                               :slice_end-slice_start] = \
                    datacost[row_start:row_end,
                             col_start:col_end,
                             slice_start:slice_end]

                # Copy the random crop of the groundtruth.
                batch_groundtruth[i,
                                  :row_end-row_start,
                                  :col_end-col_start,
                                  :slice_end-slice_start] = \
                    groundtruth[row_start:row_end,
                                col_start:col_end,
                                slice_start:slice_end]

                # Randomly rotate around z-axis.
                num_rot90 = np.random.randint(4)
                if num_rot90 > 0:
                    batch_datacost[i] = np.rot90(batch_datacost[i],
                                                 k=num_rot90,
                                                 axes=(0, 1))
                    batch_groundtruth[i] = np.rot90(batch_groundtruth[i],
                                                    k=num_rot90,
                                                    axes=(0, 1))

                # Randomly flip along x and y axis.
                flip_axis = np.random.randint(3)
                if flip_axis == 0 or flip_axis == 1:
                    batch_datacost[i] = np.flip(batch_datacost[i],
                                                axis=flip_axis)
                    batch_groundtruth[i] = np.flip(batch_groundtruth[i],
                                                   axis=flip_axis)

            yield (batch_datacost[:len(batch_idxs)],
                   batch_groundtruth[:len(batch_idxs)])

        npasses += 1

        if epoch_npasses > 0 and npasses >= epoch_npasses:
            npasses = 0
            yield
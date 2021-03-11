# Learned Semantic 3D reconstruction - ECCV 2018
Code used for the paper [Learning Priors For Semantic 3D Reconstruction](http://people.inf.ethz.ch/cian/Publications/Papers/Cherabier-Schoenberger-al-ECCV-2018.pdf).
Was implemented with Tensorflow 1.4, but also runs with TF 1.14.
The code is a joint work of Ian Cherabier and Johannes Schönberger.

## Data
The data used for the paper can be generated using the script `generate_data.sh`.
The paths need to be adjusted:
* `SCANNET_SCENES_PATH` : path to the folder containing all scannet scenes
* `SCANNET_PACKAGE_PATH`: path to the code that comes with [Scannet](https://github.com/ScanNet/ScanNet)
* `CODE_PATH`: path to the learned semantic 3d code

Normally the paths to the new generated data should still be consistent with what the training / evaluation code expects. However I did not have time to check properly if this were the case.

## Training
In order to train the model, you can use `train_scannet_final.py`.

```shell
 python train_scannet_final.py --scene_path path/to/scannet/scenes/ 
                               --scene_train_list_path path/to/scannet/scenes/train.txt 
                               --scene_val_list_path path/to/scannet/scenes/val.txt
                               --model_path path/where/to/store/log/and/model/weights
```

where val.txt and train.txt contain the name of the scenes used for validation and training.
```shell
scene0191_00
scene0191_01
scene0191_02
scene0119_00
...
```

There are more parameters that you can pass. The following are directly linked to the architecture:
* `--niter`: the number of primal dual iterations in the model
* `--nlevels`: the number of levels if you do multi-scale (`--nlevel` 1 if you only do one scale)
* `--sig`, `--tau`, `--lam` primal dual and energy hyper parameters

The results are stored in the folder specified in `--model_path`.
```shell
model_path
   |-- logs
   |-- ckpts
```
Logs contain tensorboard visualization and ckpts contains the checkpoints to restore the model.

## Evaluation
To evaluate a model you can use `eval_scannet.py`.

```shell
 python eval_scannet.py --checkpoint_path Model/ckpts/ 
                        --datacost_path path/to/one/scene/datacost.npz
                        --output_path path/to/output 
                        --label_map_path path/to/labels.txt
```

This code takes one scene and generate a reconstruction. It saves a mesh per class as well as the whole multi-class voxel grid in `probs.npz`.
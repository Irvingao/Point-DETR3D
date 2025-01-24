# Point-DETR3D

This repo implements the paper Point-DETR3D: Leveraging Imagery Data with Spatial Point Prior for Weakly Semi-Supervised 3D Object Detection.

We built our implementation upon MMdetection3D 0.17.3. The major part of the code is in the directory `project`.

## Environment

### Prerequisite

<ol>
<li> mmcv-full>=1.3.9, <=1.3.13 </li>
<li> mmdet>=2.14.0, <=2.24.0</li>
<li> mmseg>=0.14.0, <=0.20.0</li>
<li> nuscenes-devkit</li>
</ol>

### Installation

There is no neccesary to install mmdet3d separately, please install based on this repo:

```bash
git clone https://github.com/open-mmlab/mmdetection3d.git
git checkout v0.17.3
cd point-detr3d
pip3 install -v -e .
```

### Data

#### 1. General data preparation

Please follow the mmdet3d to process the data. [mmdet3d_nuscenes_guidance](https://github.com/open-mmlab/mmdetection3d/blob/main/docs/en/advanced_guides/datasets/nuscenes.md)

#### 2. split datasets with different ratio

For example, split `10%` fully-labeled data by running the following command:

```bash
python tools/split_dbinfo_data.py --split_ratio 0.1
```

## Train

Our framework consists of teacher models and student models, where we train them step by step.

For example, to train a voxel-based Point-DETR3D teacher and student model on 8 GPUs, please follow the instruction below:

### 1. Train a point-to-box teacher model

- (1) Train CenterPoint with the corrsponding training set: (`sp` means training without velocity supervisions.)

```bash
bash tools/dist_train.sh projects/configs/wss_sp_ctpt/centerpoint_01voxel_second_secfpn_4x8_cyclic_20e_nus_sp_wovel_01rate.py 8
```

- (2) Prepare pretrained backbone weights to `ckpts/`:
    - image backbone: [FCOS3D_r50](download_link)
    - lidar backbone: CenterPoint (Trained at step (1).)

fuse pretrained weights by running:

```bash
python tools/fuse_model.py --img <cam checkpoint path> --lidar <lidar checkpoint path> --out <out model path>
```

- (3) Training:

```bash
bash tools/dist_train.sh projects/configs/point3d_tec/tec_voxel_base_01rate.py 8
```

### 2. Generate pseudo labels for weakly-labeled scenes

- (1) Infer the whole train dataset for generating pseudo labels

```bash
bash tools/slurm_test.sh projects/configs/point3d_tec/tec_voxel_base_01rate.py \
    work_dirs/point3d_tec/tec_voxel_base_01rate/latest.pth \
    --out work_dirs/point3d_tec/tec_voxel_base_01rate/pseudo_res.pkl
```

- (2) Combine the psuedo labels and fully-labeled data for the student training:

```bash
python tools/generate_pseudo_labels.py data/nuscenes/nuscenes_infos_train.pkl \
    work_dirs/point3d_tec/tec_voxel_base_01rate/pseudo_res.pkl \
    work_dirs/point3d_tec/tec_voxel_base_01rate --ratio 0.1
```

### 3. Train a student model

```bash
bash tools/dist_train.sh projects/configs/point3d_stu/stu_tec_ours_centerpoint_voxel_01rate.py 8
```

## Evalation

To test, use the following command:

```bash
tools/dist_test.sh projects/configs/point3d_stu/stu_tec_ours_centerpoint_voxel_01rate.py /path/to/ckpt 8 --eval bbox
```

### Evalate detection performence by the distance range

change the the distance range by `fix_det_range`:

```bash
python tools/test_eval.py projects/configs/point3d_tec/tec_voxel_base_01rate.py work_dirs/point3d_tec/tec_voxel_base_01rate/latest.pth --eval bbox --fix_det_range 0 10
```

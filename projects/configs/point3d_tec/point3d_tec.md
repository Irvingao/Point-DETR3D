## Point Teacher

```bash
# voxel ours
tec_voxel_ours_02rate
tec_voxel_ours_05rate
GPUS=8 GPUS_PER_NODE=8 CPUS_PER_TASK=16 sh ./tools/slurm_train.sh AD_GVT_A100_40G tec_1 \
    projects/configs/point3d_tec/tec_voxel_ours_02rate.py \
    /mnt/lustrenew/gaohongzhi/work_dirs/point_objDGCNN/point3d_tec/tec_voxel_ours_02rate/

GPUS=8 GPUS_PER_NODE=8 CPUS_PER_TASK=16 sh ./tools/slurm_train.sh AD_GVT_A100_40G tec_2 \
    projects/configs/point3d_tec/tec_voxel_ours_05rate.py \
    /mnt/lustrenew/gaohongzhi/work_dirs/point_objDGCNN/point3d_tec/tec_voxel_ours_05rate/


GPUS=8 GPUS_PER_NODE=8 CPUS_PER_TASK=16 sh ./tools/slurm_train.sh AD_GVT_A100_40G tec_1 \
    projects/configs/point3d_tec/tec_voxel_ours_01rate.py \
    /mnt/lustrenew/gaohongzhi/work_dirs/point_objDGCNN/point3d_tec/tec_voxel_ours_01rate/

GPUS=8 GPUS_PER_NODE=8 CPUS_PER_TASK=16 sh ./tools/slurm_train.sh AD_GVT_A100_40G tec_2 \
    projects/configs/point3d_tec/tec_voxel_ours_005rate.py \
    /mnt/lustrenew/gaohongzhi/work_dirs/point_objDGCNN/point3d_tec/tec_voxel_ours_005rate/

GPUS=8 GPUS_PER_NODE=8 CPUS_PER_TASK=16 sh ./tools/slurm_train.sh AD_GVT_A100_40G tec_3 \
    projects/configs/point3d_tec/tec_voxel_ours_002rate.py \
    /mnt/lustrenew/gaohongzhi/work_dirs/point_objDGCNN/point3d_tec/tec_voxel_ours_002rate/


# voxel base
tec_voxel_base_02rate
tec_voxel_base_05rate
GPUS=8 GPUS_PER_NODE=8 CPUS_PER_TASK=16 sh ./tools/slurm_train.sh AD_GVT_A100_40G tec2_1 \
    projects/configs/point3d_tec/tec_voxel_base_05rate.py \
    /mnt/lustrenew/gaohongzhi/work_dirs/point_objDGCNN/point3d_tec/tec_voxel_base_05rate/

GPUS=8 GPUS_PER_NODE=8 CPUS_PER_TASK=16 sh ./tools/slurm_train.sh AD_GVT_A100_40G tec2_2 \
    projects/configs/point3d_tec/tec_voxel_base_02rate.py \
    /mnt/lustrenew/gaohongzhi/work_dirs/point_objDGCNN/point3d_tec/tec_voxel_base_02rate/

GPUS=8 GPUS_PER_NODE=8 CPUS_PER_TASK=16 sh ./tools/slurm_train.sh AD_GVT_A100_40G tec2_1 \
    projects/configs/point3d_tec/tec_voxel_base_01rate.py \
    /mnt/lustrenew/gaohongzhi/work_dirs/point_objDGCNN/point3d_tec/tec_voxel_base_01rate/

GPUS=8 GPUS_PER_NODE=8 CPUS_PER_TASK=16 sh ./tools/slurm_train.sh AD_GVT_A100_40G tec2_2 \
    projects/configs/point3d_tec/tec_voxel_base_005rate.py \
    /mnt/lustrenew/gaohongzhi/work_dirs/point_objDGCNN/point3d_tec/tec_voxel_base_005rate/

GPUS=8 GPUS_PER_NODE=8 CPUS_PER_TASK=16 sh ./tools/slurm_train.sh AD_GVT_A100_40G tec2_3 \
    projects/configs/point3d_tec/tec_voxel_base_002rate.py \
    /mnt/lustrenew/gaohongzhi/work_dirs/point_objDGCNN/point3d_tec/tec_voxel_base_002rate/


# pillar ours

GPUS=8 GPUS_PER_NODE=8 CPUS_PER_TASK=16 sh ./tools/slurm_train.sh AD_GVT_A100_40G pillar_1 \
    projects/configs/point3d_tec/tec_pillar_ours_01rate.py \
    /mnt/lustrenew/gaohongzhi/work_dirs/point_objDGCNN/point3d_tec/tec_pillar_ours_01rate/

GPUS=8 GPUS_PER_NODE=8 CPUS_PER_TASK=16 sh ./tools/slurm_train.sh AD_GVT_A100_40G pillar_2 \
    projects/configs/point3d_tec/tec_pillar_ours_005rate.py \
    /mnt/lustrenew/gaohongzhi/work_dirs/point_objDGCNN/point3d_tec/tec_pillar_ours_005rate/

GPUS=8 GPUS_PER_NODE=8 CPUS_PER_TASK=16 sh ./tools/slurm_train.sh AD_GVT_A100_40G pillar_3 \
    projects/configs/point3d_tec/tec_pillar_ours_002rate.py \
    /mnt/lustrenew/gaohongzhi/work_dirs/point_objDGCNN/point3d_tec/tec_pillar_ours_002rate/


# pillar base

GPUS=8 GPUS_PER_NODE=8 CPUS_PER_TASK=16 sh ./tools/slurm_train.sh AD_GVT_A100_40G pillar2_1 \
    projects/configs/point3d_tec/tec_pillar_base_01rate.py \
    /mnt/lustrenew/gaohongzhi/work_dirs/point_objDGCNN/point3d_tec/tec_pillar_base_01rate/

GPUS=8 GPUS_PER_NODE=8 CPUS_PER_TASK=16 sh ./tools/slurm_train.sh AD_GVT_A100_40G pillar2_2 \
    projects/configs/point3d_tec/tec_pillar_base_005rate.py \
    /mnt/lustrenew/gaohongzhi/work_dirs/point_objDGCNN/point3d_tec/tec_pillar_base_005rate/

GPUS=8 GPUS_PER_NODE=8 CPUS_PER_TASK=16 sh ./tools/slurm_train.sh AD_GVT_A100_40G pillar2_3 \
    projects/configs/point3d_tec/tec_pillar_base_002rate.py \
    /mnt/lustrenew/gaohongzhi/work_dirs/point_objDGCNN/point3d_tec/tec_pillar_base_002rate/

GPUS=4 GPUS_PER_NODE=4 CPUS_PER_TASK=16 sh ./tools/slurm_test.sh AD_GVT_A100_40G tec_res \
    projects/configs/point3d_tec/tec_voxel_ours_05rate.py \
    /mnt/lustrenew/gaohongzhi/work_dirs/point_objDGCNN/point3d_tec/tec_voxel_ours_05rate/latest.pth \
    --out /mnt/lustrenew/gaohongzhi/work_dirs/point_objDGCNN/point3d_tec/tec_voxel_ours_05rate/tec_voxel05_res.pkl
```

### 0. 先新建py文件，后缀加_gen_train_pkl，并修改config中test dataset的ann_file，将val.pkl为 nuscenes_infos_train.pkl


### 1. 生成 pseudo pkl
```bash
# √
tec_voxel_ours_01rate_pkl tec_voxel_ours_01rate
tec_voxel_ours_005rate_pkl tec_voxel_ours_005rate
tec_voxel_ours_002rate_pkl tec_voxel_ours_002rate

# √
tec_pillar_base_01rate_pkl tec_pillar_base_01rate
tec_pillar_base_005rate_pkl tec_pillar_base_005rate
tec_pillar_base_002rate_pkl tec_pillar_base_002rate

# 
tec_pillar_ours_01rate_pkl tec_pillar_ours_01rate
tec_pillar_ours_005rate_pkl tec_pillar_ours_005rate
tec_pillar_ours_002rate_pkl tec_pillar_ours_002rate


GPUS=8 GPUS_PER_NODE=8 CPUS_PER_TASK=16 sh ./tools/slurm_test.sh AD_GVT_A100_40G tec_res_pkl \
    projects/configs/point3d_tec/tec_pillar_ours_01rate_pkl.py \
    /mnt/lustrenew/gaohongzhi/work_dirs/point_objDGCNN/point3d_tec/tec_pillar_ours_01rate/latest.pth \
    --out /mnt/lustrenew/gaohongzhi/work_dirs/point_objDGCNN/point3d_tec/tec_pillar_ours_01rate/pseudo_res.pkl

GPUS=8 GPUS_PER_NODE=8 CPUS_PER_TASK=16 sh ./tools/slurm_test.sh AD_GVT_A100_40G tec_res_pkl_2 \
    projects/configs/point3d_tec/tec_pillar_ours_005rate_pkl.py \
    /mnt/lustrenew/gaohongzhi/work_dirs/point_objDGCNN/point3d_tec/tec_pillar_ours_005rate/latest.pth \
    --out /mnt/lustrenew/gaohongzhi/work_dirs/point_objDGCNN/point3d_tec/tec_pillar_ours_005rate/pseudo_res.pkl

GPUS=8 GPUS_PER_NODE=8 CPUS_PER_TASK=16 sh ./tools/slurm_test.sh AD_GVT_A100_40G tec_res_pkl_3 \
    projects/configs/point3d_tec/tec_pillar_ours_002rate_pkl.py \
    /mnt/lustrenew/gaohongzhi/work_dirs/point_objDGCNN/point3d_tec/tec_pillar_ours_002rate/latest.pth \
    --out /mnt/lustrenew/gaohongzhi/work_dirs/point_objDGCNN/point3d_tec/tec_pillar_ours_002rate/pseudo_res.pkl

tec_voxel_base_02rate
tec_voxel_base_02rate_pkl

tec_voxel_ours_02rate
tec_voxel_ours_02rate_pkl

tec_voxel_base_05rate
tec_voxel_base_05rate_pkl

tec_voxel_ours_05rate
tec_voxel_ours_05rate_pkl
GPUS=4 GPUS_PER_NODE=4 CPUS_PER_TASK=16 sh ./tools/slurm_test.sh AD_GVT_A100_40G tec_res_pkl1 \
    projects/configs/point3d_tec/tec_voxel_ours_05rate_pkl.py \
    /mnt/lustrenew/gaohongzhi/work_dirs/point_objDGCNN/point3d_tec/tec_voxel_ours_05rate/latest.pth \
    --out /mnt/lustrenew/gaohongzhi/work_dirs/point_objDGCNN/point3d_tec/tec_voxel_ours_05rate/pseudo_res.pkl
```

### 2. 生成pseudo labels+gt的混合pkl
```bash
tec_voxel_base_02rate
tec_voxel_ours_02rate
tec_voxel_base_05rate
tec_voxel_ours_05rate
srun -p AD_GVT_A100_40G --gres=gpu:1 -n1 --ntasks-per-node=1 --cpus-per-task=16 --quotatype=auto --job-name=gen_pseudo_label_1 \
    python tools/generate_pseudo_labels.py data/nuscenes/nuscenes_infos_train.pkl \
    /mnt/lustrenew/gaohongzhi/work_dirs/point_objDGCNN/point3d_tec/tec_voxel_ours_05rate/pseudo_res.pkl \
    /mnt/lustrenew/gaohongzhi/work_dirs/point_objDGCNN/point3d_tec/tec_voxel_ours_05rate --ratio 0.5




srun -p AD_GVT_A100_40G --gres=gpu:1 -n1 --ntasks-per-node=1 --cpus-per-task=8 --quotatype=auto --job-name=gen_pseudo_label_1 \
    python tools/generate_pseudo_labels.py data/nuscenes/nuscenes_infos_train.pkl \
    /mnt/lustrenew/gaohongzhi/work_dirs/point_objDGCNN/point3d_tec/tec_pillar_ours_01rate/pseudo_res.pkl \
    /mnt/lustrenew/gaohongzhi/work_dirs/point_objDGCNN/point3d_tec/tec_pillar_ours_01rate --ratio 0.1

srun -p AD_GVT_A100_40G --gres=gpu:1 -n1 --ntasks-per-node=1 --cpus-per-task=8 --quotatype=auto --job-name=gen_pseudo_label_2 \
    python tools/generate_pseudo_labels.py data/nuscenes/nuscenes_infos_train.pkl \
    /mnt/lustrenew/gaohongzhi/work_dirs/point_objDGCNN/point3d_tec/tec_pillar_ours_005rate/pseudo_res.pkl \
    /mnt/lustrenew/gaohongzhi/work_dirs/point_objDGCNN/point3d_tec/tec_pillar_ours_005rate --ratio 0.05

srun -p AD_GVT_A100_40G --gres=gpu:1 -n1 --ntasks-per-node=1 --cpus-per-task=8 --quotatype=auto --job-name=gen_pseudo_label_3 \
    python tools/generate_pseudo_labels.py data/nuscenes/nuscenes_infos_train.pkl \
    /mnt/lustrenew/gaohongzhi/work_dirs/point_objDGCNN/point3d_tec/tec_pillar_ours_002rate/pseudo_res.pkl \
    /mnt/lustrenew/gaohongzhi/work_dirs/point_objDGCNN/point3d_tec/tec_pillar_ours_002rate --ratio 0.02




srun -p AD_GVT_A100_40G --gres=gpu:1 -n1 --ntasks-per-node=1 --cpus-per-task=8 --quotatype=auto --job-name=gen_pseudo_label_1 \
    python tools/generate_pseudo_labels.py data/nuscenes/nuscenes_infos_train.pkl \
    /mnt/lustrenew/gaohongzhi/work_dirs/point_objDGCNN/point3d_tec/tec_voxel_ours_01rate/pseudo_res.pkl \
    /mnt/lustrenew/gaohongzhi/work_dirs/point_objDGCNN/point3d_tec/tec_voxel_ours_01rate --ratio 0.1

srun -p AD_GVT_A100_40G --gres=gpu:1 -n1 --ntasks-per-node=1 --cpus-per-task=8 --quotatype=auto --job-name=gen_pseudo_label_2 \
    python tools/generate_pseudo_labels.py data/nuscenes/nuscenes_infos_train.pkl \
    /mnt/lustrenew/gaohongzhi/work_dirs/point_objDGCNN/point3d_tec/tec_voxel_ours_005rate/pseudo_res.pkl \
    /mnt/lustrenew/gaohongzhi/work_dirs/point_objDGCNN/point3d_tec/tec_voxel_ours_005rate --ratio 0.05

srun -p AD_GVT_A100_40G --gres=gpu:1 -n1 --ntasks-per-node=1 --cpus-per-task=8 --quotatype=auto --job-name=gen_pseudo_label_3 \
    python tools/generate_pseudo_labels.py data/nuscenes/nuscenes_infos_train.pkl \
    /mnt/lustrenew/gaohongzhi/work_dirs/point_objDGCNN/point3d_tec/tec_voxel_ours_002rate/pseudo_res.pkl \
    /mnt/lustrenew/gaohongzhi/work_dirs/point_objDGCNN/point3d_tec/tec_voxel_ours_002rate --ratio 0.02
```



## objdgcnn
```bash
GPUS=8 GPUS_PER_NODE=8 CPUS_PER_TASK=16 sh ./tools/slurm_train.sh AD_GVT_A100_40G obj_1 \
    projects/configs/point3d_tec/objdgcnn_01rate.py \
    /mnt/lustrenew/gaohongzhi/work_dirs/point_objDGCNN/point3d_tec/objdgcnn_01rate/

GPUS=8 GPUS_PER_NODE=8 CPUS_PER_TASK=16 sh ./tools/slurm_train.sh AD_GVT_A100_40G obj_2 \
    projects/configs/point3d_tec/objdgcnn_voxel_005rate.py \
    /mnt/lustrenew/gaohongzhi/work_dirs/point_objDGCNN/point3d_tec/objdgcnn_voxel_005rate/

GPUS=8 GPUS_PER_NODE=8 CPUS_PER_TASK=16 sh ./tools/slurm_train.sh AD_GVT_A100_40G obj_3 \
    projects/configs/point3d_tec/objdgcnn_voxel_002rate.py \
    /mnt/lustrenew/gaohongzhi/work_dirs/point_objDGCNN/point3d_tec/objdgcnn_voxel_002rate/

GPUS=8 GPUS_PER_NODE=8 CPUS_PER_TASK=16 sh ./tools/slurm_train.sh AD_GVT_A100_40G obj_4 \
    projects/configs/point3d_tec/objdgcnn_voxel_1rate.py \
    /mnt/lustrenew/gaohongzhi/work_dirs/point_objDGCNN/point3d_tec/objdgcnn_voxel_1rate/

# pillar
GPUS=8 GPUS_PER_NODE=8 CPUS_PER_TASK=16 sh ./tools/slurm_train.sh AD_GVT_A100_40G obj2_1 \
    projects/configs/point3d_tec/objdgcnn_pillar_01rate.py \
    /mnt/lustrenew/gaohongzhi/work_dirs/point_objDGCNN/point3d_tec/objdgcnn_pillar_01rate/

GPUS=8 GPUS_PER_NODE=8 CPUS_PER_TASK=16 sh ./tools/slurm_train.sh AD_GVT_A100_40G obj2_2 \
    projects/configs/point3d_tec/objdgcnn_pillar_005rate.py \
    /mnt/lustrenew/gaohongzhi/work_dirs/point_objDGCNN/point3d_tec/objdgcnn_pillar_005rate/

GPUS=8 GPUS_PER_NODE=8 CPUS_PER_TASK=16 sh ./tools/slurm_train.sh AD_GVT_A100_40G obj2_3 \
    projects/configs/point3d_tec/objdgcnn_pillar_002rate.py \
    /mnt/lustrenew/gaohongzhi/work_dirs/point_objDGCNN/point3d_tec/objdgcnn_pillar_002rate/

GPUS=8 GPUS_PER_NODE=8 CPUS_PER_TASK=16 sh ./tools/slurm_train.sh AD_GVT_A100_40G obj2_4 \
    projects/configs/point3d_tec/objdgcnn_pillar_1rate.py \
    /mnt/lustrenew/gaohongzhi/work_dirs/point_objDGCNN/point3d_tec/objdgcnn_pillar_1rate/

objdgcnn_pillar_002rate
objdgcnn_pillar_005rate
objdgcnn_pillar_01rate
objdgcnn_pillar_1rate

```


## test range

```bash
tec_voxel_ours_01rate
tec_voxel_base_01rate

# 1. 生成pkl
GPUS=8 GPUS_PER_NODE=8 CPUS_PER_TASK=32 sh ./tools/slurm_test_range.sh AD_GVT_A100_40G tec_res_pkl \
    projects/configs/point3d_tec/tec_voxel_base_01rate.py  \
    /mnt/lustrenew/gaohongzhi/work_dirs/point_objDGCNN/point3d_tec/tec_voxel_base_01rate/latest.pth \
    --out /mnt/lustrenew/gaohongzhi/work_dirs/point_objDGCNN/point3d_tec/tec_voxel_base_01rate/res.pkl

# 2. 更改range setting eval (自动检查是否存在res.pkl，如果存在则不用infer)
GPUS=1 GPUS_PER_NODE=1 CPUS_PER_TASK=32 sh ./tools/slurm_test_range.sh AD_GVT_A100_40G test_range_1 \
    /mnt/lustrenew/gaohongzhi/work_dirs/point_objDGCNN/point3d_tec/tec_voxel_base_01rate/tec_voxel_base_01rate.py \
    /mnt/lustrenew/gaohongzhi/work_dirs/point_objDGCNN/point3d_tec/tec_voxel_base_01rate/latest.pth --eval bbox --fix_det_range 0 10

GPUS=1 GPUS_PER_NODE=1 CPUS_PER_TASK=32 sh ./tools/slurm_test_range.sh AD_GVT_A100_40G test_range_2 \
    /mnt/lustrenew/gaohongzhi/work_dirs/point_objDGCNN/point3d_tec/tec_voxel_base_01rate/tec_voxel_base_01rate.py \
    /mnt/lustrenew/gaohongzhi/work_dirs/point_objDGCNN/point3d_tec/tec_voxel_base_01rate/latest.pth --eval bbox --fix_det_range 10 20

GPUS=1 GPUS_PER_NODE=1 CPUS_PER_TASK=32 sh ./tools/slurm_test_range.sh AD_GVT_A100_40G test_range_3 \
    /mnt/lustrenew/gaohongzhi/work_dirs/point_objDGCNN/point3d_tec/tec_voxel_base_01rate/tec_voxel_base_01rate.py \
    /mnt/lustrenew/gaohongzhi/work_dirs/point_objDGCNN/point3d_tec/tec_voxel_base_01rate/latest.pth --eval bbox --fix_det_range 20 30

GPUS=1 GPUS_PER_NODE=1 CPUS_PER_TASK=32 sh ./tools/slurm_test_range.sh AD_GVT_A100_40G test_range_4 \
    /mnt/lustrenew/gaohongzhi/work_dirs/point_objDGCNN/point3d_tec/tec_voxel_base_01rate/tec_voxel_base_01rate.py \
    /mnt/lustrenew/gaohongzhi/work_dirs/point_objDGCNN/point3d_tec/tec_voxel_base_01rate/latest.pth --eval bbox --fix_det_range 30 40

GPUS=1 GPUS_PER_NODE=1 CPUS_PER_TASK=32 sh ./tools/slurm_test_range.sh AD_GVT_A100_40G test_range_5 \
    /mnt/lustrenew/gaohongzhi/work_dirs/point_objDGCNN/point3d_tec/tec_voxel_base_01rate/tec_voxel_base_01rate.py \
    /mnt/lustrenew/gaohongzhi/work_dirs/point_objDGCNN/point3d_tec/tec_voxel_base_01rate/latest.pth --eval bbox --fix_det_range 40 52


```

```bash



# base
# voxel 

stu_tec_base_centerpoint_voxel_01rate
stu_tec_base_centerpoint_voxel_005rate
stu_tec_base_centerpoint_voxel_002rate
GPUS=8 GPUS_PER_NODE=8 CPUS_PER_TASK=16 sh ./tools/slurm_train.sh AD_GVT_A100_40G tec_1 \
    projects/configs/point3d_stu/stu_tec_base_centerpoint_voxel_01rate.py \
    /mnt/lustrenew/gaohongzhi/work_dirs/point_objDGCNN/point3d_stu/stu_tec_base_centerpoint_voxel_01rate/

GPUS=8 GPUS_PER_NODE=8 CPUS_PER_TASK=16 sh ./tools/slurm_train.sh AD_GVT_A100_40G tec_2 \
    projects/configs/point3d_stu/stu_tec_base_centerpoint_voxel_005rate.py \
    /mnt/lustrenew/gaohongzhi/work_dirs/point_objDGCNN/point3d_stu/stu_tec_base_centerpoint_voxel_005rate/

GPUS=8 GPUS_PER_NODE=8 CPUS_PER_TASK=16 sh ./tools/slurm_train.sh AD_GVT_A100_40G tec_3 \
    projects/configs/point3d_stu/stu_tec_base_centerpoint_voxel_002rate.py \
    /mnt/lustrenew/gaohongzhi/work_dirs/point_objDGCNN/point3d_stu/stu_tec_base_centerpoint_voxel_002rate/

# 20%
stu_tec_base_centerpoint_voxel_02rate
GPUS=8 GPUS_PER_NODE=8 CPUS_PER_TASK=16 sh ./tools/slurm_train_reserved.sh AD_GVT_A100_40G tec_3 \
    projects/configs/point3d_stu/stu_tec_base_centerpoint_voxel_02rate.py \
    /mnt/lustrenew/gaohongzhi/work_dirs/point_objDGCNN/point3d_stu/stu_tec_base_centerpoint_voxel_02rate/ \
    --resume-from /mnt/lustrenew/gaohongzhi/work_dirs/point_objDGCNN/point3d_stu/stu_tec_base_centerpoint_voxel_02rate/latest.pth
stu_tec_ours_centerpoint_voxel_02rate
GPUS=8 GPUS_PER_NODE=8 CPUS_PER_TASK=16 sh ./tools/slurm_train.sh AD_GVT_A100_40G tec_4 \
    projects/configs/point3d_stu/stu_tec_ours_centerpoint_voxel_02rate.py \
    /mnt/lustrenew/gaohongzhi/work_dirs/point_objDGCNN/point3d_stu/stu_tec_ours_centerpoint_voxel_02rate/ \
    --resume-from /mnt/lustrenew/gaohongzhi/work_dirs/point_objDGCNN/point3d_stu/stu_tec_ours_centerpoint_voxel_02rate/latest.pth
stu_tec_base_centerpoint_voxel_05rate
GPUS=8 GPUS_PER_NODE=8 CPUS_PER_TASK=16 sh ./tools/slurm_train_reserved.sh AD_GVT_A100_40G tec_5 \
    projects/configs/point3d_stu/stu_tec_base_centerpoint_voxel_05rate.py \
    /mnt/lustrenew/gaohongzhi/work_dirs/point_objDGCNN/point3d_stu/stu_tec_base_centerpoint_voxel_05rate/ \
    --resume-from /mnt/lustrenew/gaohongzhi/work_dirs/point_objDGCNN/point3d_stu/stu_tec_base_centerpoint_voxel_05rate/latest.pth
stu_tec_ours_centerpoint_voxel_05rate
GPUS=8 GPUS_PER_NODE=8 CPUS_PER_TASK=16 sh ./tools/slurm_train_reserved.sh AD_GVT_A100_40G tec_6 \
    projects/configs/point3d_stu/stu_tec_ours_centerpoint_voxel_05rate.py \
    /mnt/lustrenew/gaohongzhi/work_dirs/point_objDGCNN/point3d_stu/stu_tec_ours_centerpoint_voxel_05rate/ \
    --resume-from /mnt/lustrenew/gaohongzhi/work_dirs/point_objDGCNN/point3d_stu/stu_tec_ours_centerpoint_voxel_05rate/latest.pth


# pillar
stu_tec_base_centerpoint_pillar_01rate
stu_tec_base_centerpoint_pillar_005rate
stu_tec_base_centerpoint_pillar_002rate
GPUS=8 GPUS_PER_NODE=8 CPUS_PER_TASK=16 sh ./tools/slurm_train.sh AD_GVT_A100_40G tec_4 \
    projects/configs/point3d_stu/stu_tec_base_centerpoint_pillar_01rate.py \
    /mnt/lustrenew/gaohongzhi/work_dirs/point_objDGCNN/point3d_stu/stu_tec_base_centerpoint_pillar_01rate/ \
    --resume-from /mnt/lustrenew/gaohongzhi/work_dirs/point_objDGCNN/point3d_stu/stu_tec_base_centerpoint_pillar_01rate/latest.pth

GPUS=8 GPUS_PER_NODE=8 CPUS_PER_TASK=16 sh ./tools/slurm_train.sh AD_GVT_A100_40G tec_5 \
    projects/configs/point3d_stu/stu_tec_base_centerpoint_pillar_005rate.py \
    /mnt/lustrenew/gaohongzhi/work_dirs/point_objDGCNN/point3d_stu/stu_tec_base_centerpoint_pillar_005rate/ \
    --resume-from /mnt/lustrenew/gaohongzhi/work_dirs/point_objDGCNN/point3d_stu/stu_tec_base_centerpoint_pillar_005rate/latest.pth

GPUS=8 GPUS_PER_NODE=8 CPUS_PER_TASK=16 sh ./tools/slurm_train.sh AD_GVT_A100_40G tec_6 \
    projects/configs/point3d_stu/stu_tec_base_centerpoint_pillar_002rate.py \
    /mnt/lustrenew/gaohongzhi/work_dirs/point_objDGCNN/point3d_stu/stu_tec_base_centerpoint_pillar_002rate/ \
    --resume-from /mnt/lustrenew/gaohongzhi/work_dirs/point_objDGCNN/point3d_stu/stu_tec_base_centerpoint_pillar_002rate/latest.pth

# ours

# voxel 
stu_tec_ours_centerpoint_voxel_01rate
stu_tec_ours_centerpoint_voxel_005rate
stu_tec_ours_centerpoint_voxel_002rate
# slurm_train_reserved
GPUS=8 GPUS_PER_NODE=8 CPUS_PER_TASK=16 sh ./tools/slurm_train_reserved.sh AD_GVT_A100_40G ours_1 \
    projects/configs/point3d_stu/stu_tec_ours_centerpoint_voxel_01rate.py \
    /mnt/lustrenew/gaohongzhi/work_dirs/point_objDGCNN/point3d_stu/stu_tec_ours_centerpoint_voxel_01rate/ \
    --resume-from /mnt/lustrenew/gaohongzhi/work_dirs/point_objDGCNN/point3d_stu/stu_tec_ours_centerpoint_voxel_01rate/latest.pth

GPUS=8 GPUS_PER_NODE=8 CPUS_PER_TASK=16 sh ./tools/slurm_train_reserved.sh AD_GVT_A100_40G ours_2 \
    projects/configs/point3d_stu/stu_tec_ours_centerpoint_voxel_005rate.py \
    /mnt/lustrenew/gaohongzhi/work_dirs/point_objDGCNN/point3d_stu/stu_tec_ours_centerpoint_voxel_005rate/ \
    --resume-from /mnt/lustrenew/gaohongzhi/work_dirs/point_objDGCNN/point3d_stu/stu_tec_ours_centerpoint_voxel_005rate/latest.pth

GPUS=8 GPUS_PER_NODE=8 CPUS_PER_TASK=16 sh ./tools/slurm_train_reserved.sh AD_GVT_A100_40G ours_3 \
    projects/configs/point3d_stu/stu_tec_ours_centerpoint_voxel_002rate.py \
    /mnt/lustrenew/gaohongzhi/work_dirs/point_objDGCNN/point3d_stu/stu_tec_ours_centerpoint_voxel_002rate/ \
    --resume-from /mnt/lustrenew/gaohongzhi/work_dirs/point_objDGCNN/point3d_stu/stu_tec_ours_centerpoint_voxel_002rate/latest.pth

stu_tec_ours_centerpoint_voxel_005rate_pointSF4
stu_tec_ours_centerpoint_voxel_002rate_pointSF4
GPUS=8 GPUS_PER_NODE=8 CPUS_PER_TASK=16 sh ./tools/slurm_train_reserved.sh AD_GVT_A100_40G ourssf_1 \
    projects/configs/point3d_stu/stu_tec_ours_centerpoint_voxel_01rate.py \
    /mnt/lustrenew/gaohongzhi/work_dirs/point_objDGCNN/point3d_stu/stu_tec_ours_centerpoint_voxel_01rate/ \
    --resume-from /mnt/lustrenew/gaohongzhi/work_dirs/point_objDGCNN/point3d_stu/stu_tec_ours_centerpoint_voxel_01rate/latest.pth

GPUS=8 GPUS_PER_NODE=8 CPUS_PER_TASK=16 sh ./tools/slurm_train_reserved.sh AD_GVT_A100_40G ourssf_2 \
    projects/configs/point3d_stu/stu_tec_ours_centerpoint_voxel_005rate_pointSF4.py \
    /mnt/lustrenew/gaohongzhi/work_dirs/point_objDGCNN/point3d_stu/stu_tec_ours_centerpoint_voxel_005rate_pointSF4/ \
    --resume-from /mnt/lustrenew/gaohongzhi/work_dirs/point_objDGCNN/point3d_stu/stu_tec_ours_centerpoint_voxel_005rate_pointSF4/latest.pth

GPUS=8 GPUS_PER_NODE=8 CPUS_PER_TASK=16 sh ./tools/slurm_train_reserved.sh AD_GVT_A100_40G ourssf_3 \
    projects/configs/point3d_stu/stu_tec_ours_centerpoint_voxel_002rate_pointSF4.py \
    /mnt/lustrenew/gaohongzhi/work_dirs/point_objDGCNN/point3d_stu/stu_tec_ours_centerpoint_voxel_002rate_pointSF4/ \
    --resume-from /mnt/lustrenew/gaohongzhi/work_dirs/point_objDGCNN/point3d_stu/stu_tec_ours_centerpoint_voxel_002rate_pointSF4/latest.pth


stu_tec_ours_centerpoint_pillar_01rate
stu_tec_ours_centerpoint_pillar_005rate
stu_tec_ours_centerpoint_pillar_002rate
GPUS=8 GPUS_PER_NODE=8 CPUS_PER_TASK=16 sh ./tools/slurm_train_reserved.sh AD_GVT_A100_40G our_4 \
    projects/configs/point3d_stu/stu_tec_ours_centerpoint_pillar_01rate.py \
    /mnt/lustrenew/gaohongzhi/work_dirs/point_objDGCNN/point3d_stu/stu_tec_ours_centerpoint_pillar_01rate/ \
    --resume-from /mnt/lustrenew/gaohongzhi/work_dirs/point_objDGCNN/point3d_stu/stu_tec_ours_centerpoint_pillar_01rate/latest.pth

GPUS=8 GPUS_PER_NODE=8 CPUS_PER_TASK=16 sh ./tools/slurm_train.sh AD_GVT_A100_40G our_5 \
    projects/configs/point3d_stu/stu_tec_ours_centerpoint_pillar_005rate.py \
    /mnt/lustrenew/gaohongzhi/work_dirs/point_objDGCNN/point3d_stu/stu_tec_ours_centerpoint_pillar_005rate/ \
    --resume-from /mnt/lustrenew/gaohongzhi/work_dirs/point_objDGCNN/point3d_stu/stu_tec_ours_centerpoint_pillar_005rate/latest.pth

GPUS=8 GPUS_PER_NODE=8 CPUS_PER_TASK=16 sh ./tools/slurm_train.sh AD_GVT_A100_40G our_6 \
    projects/configs/point3d_stu/stu_tec_ours_centerpoint_pillar_002rate.py \
    /mnt/lustrenew/gaohongzhi/work_dirs/point_objDGCNN/point3d_stu/stu_tec_ours_centerpoint_pillar_002rate/ \
    --resume-from /mnt/lustrenew/gaohongzhi/work_dirs/point_objDGCNN/point3d_stu/stu_tec_ours_centerpoint_pillar_002rate/latest.pth


stu_tec_ours_centerpoint_pillar_01rate_SF4
stu_tec_ours_centerpoint_pillar_005rate_SF4
stu_tec_ours_centerpoint_pillar_002rate_SF4

GPUS=8 GPUS_PER_NODE=8 CPUS_PER_TASK=16 sh ./tools/slurm_train_reserved.sh AD_GVT_A100_40G ourssf_4 \
    projects/configs/point3d_stu/stu_tec_ours_centerpoint_pillar_01rate_SF4.py \
    /mnt/lustrenew/gaohongzhi/work_dirs/point_objDGCNN/point3d_stu/stu_tec_ours_centerpoint_pillar_01rate_SF4/ \
    --resume-from /mnt/lustrenew/gaohongzhi/work_dirs/point_objDGCNN/point3d_stu/stu_tec_ours_centerpoint_pillar_01rate_SF4/latest.pth

GPUS=8 GPUS_PER_NODE=8 CPUS_PER_TASK=16 sh ./tools/slurm_train_reserved.sh AD_GVT_A100_40G ourssf_5 \
    projects/configs/point3d_stu/stu_tec_ours_centerpoint_pillar_005rate_SF4.py \
    /mnt/lustrenew/gaohongzhi/work_dirs/point_objDGCNN/point3d_stu/stu_tec_ours_centerpoint_pillar_005rate_SF4/ \
    --resume-from /mnt/lustrenew/gaohongzhi/work_dirs/point_objDGCNN/point3d_stu/stu_tec_ours_centerpoint_pillar_005rate_SF4/latest.pth

GPUS=8 GPUS_PER_NODE=8 CPUS_PER_TASK=16 sh ./tools/slurm_train_reserved.sh AD_GVT_A100_40G ourssf_6 \
    projects/configs/point3d_stu/stu_tec_ours_centerpoint_pillar_002rate_SF4.py \
    /mnt/lustrenew/gaohongzhi/work_dirs/point_objDGCNN/point3d_stu/stu_tec_ours_centerpoint_pillar_002rate_SF4/ \
    --resume-from /mnt/lustrenew/gaohongzhi/work_dirs/point_objDGCNN/point3d_stu/stu_tec_ours_centerpoint_pillar_002rate_SF4/latest.pth

stu_tec_ours_centerpoint_pillar_01rate_voxel_tec_SF4
stu_tec_ours_centerpoint_pillar_005rate_voxel_tec_SF4
stu_tec_ours_centerpoint_pillar_002rate_voxel_tec_SF4
GPUS=8 GPUS_PER_NODE=8 CPUS_PER_TASK=16 sh ./tools/slurm_train.sh AD_GVT_A100_40G v_tec_1 \
    projects/configs/point3d_stu/stu_tec_ours_centerpoint_pillar_01rate_voxel_tec_SF4.py \
    /mnt/lustrenew/gaohongzhi/work_dirs/point_objDGCNN/point3d_stu/stu_tec_ours_centerpoint_pillar_01rate_voxel_tec_SF4/ \
    --resume-from /mnt/lustrenew/gaohongzhi/work_dirs/point_objDGCNN/point3d_stu/stu_tec_ours_centerpoint_pillar_01rate_voxel_tec_SF4/latest.pth

GPUS=8 GPUS_PER_NODE=8 CPUS_PER_TASK=16 sh ./tools/slurm_train.sh AD_GVT_A100_40G v_tec_2 \
    projects/configs/point3d_stu/stu_tec_ours_centerpoint_pillar_005rate_voxel_tec_SF4.py \
    /mnt/lustrenew/gaohongzhi/work_dirs/point_objDGCNN/point3d_stu/stu_tec_ours_centerpoint_pillar_005rate_voxel_tec_SF4/ \
    --resume-from /mnt/lustrenew/gaohongzhi/work_dirs/point_objDGCNN/point3d_stu/stu_tec_ours_centerpoint_pillar_005rate_voxel_tec_SF4/latest.pth

GPUS=8 GPUS_PER_NODE=8 CPUS_PER_TASK=16 sh ./tools/slurm_train.sh AD_GVT_A100_40G v_tec_3 \
    projects/configs/point3d_stu/stu_tec_ours_centerpoint_pillar_002rate_voxel_tec_SF4.py \
    /mnt/lustrenew/gaohongzhi/work_dirs/point_objDGCNN/point3d_stu/stu_tec_ours_centerpoint_pillar_002rate_voxel_tec_SF4/ \
    --resume-from /mnt/lustrenew/gaohongzhi/work_dirs/point_objDGCNN/point3d_stu/stu_tec_ours_centerpoint_pillar_002rate_voxel_tec_SF4/latest.pth

stu_tec_ours_centerpoint_pillar_01rate_voxel_tec
stu_tec_ours_centerpoint_pillar_005rate_voxel_tec
stu_tec_ours_centerpoint_pillar_002rate_voxel_tec
GPUS=8 GPUS_PER_NODE=8 CPUS_PER_TASK=16 sh ./tools/slurm_train.sh AD_GVT_A100_40G v_tec_1 \
    projects/configs/point3d_stu/stu_tec_ours_centerpoint_pillar_01rate_voxel_tec.py \
    /mnt/lustrenew/gaohongzhi/work_dirs/point_objDGCNN/point3d_stu/stu_tec_ours_centerpoint_pillar_01rate_voxel_tec/ \
    --resume-from /mnt/lustrenew/gaohongzhi/work_dirs/point_objDGCNN/point3d_stu/stu_tec_ours_centerpoint_pillar_01rate_voxel_tec/latest.pth

GPUS=8 GPUS_PER_NODE=8 CPUS_PER_TASK=16 sh ./tools/slurm_train.sh AD_GVT_A100_40G v_tec_2 \
    projects/configs/point3d_stu/stu_tec_ours_centerpoint_pillar_005rate_voxel_tec.py \
    /mnt/lustrenew/gaohongzhi/work_dirs/point_objDGCNN/point3d_stu/stu_tec_ours_centerpoint_pillar_005rate_voxel_tec/ \
    --resume-from /mnt/lustrenew/gaohongzhi/work_dirs/point_objDGCNN/point3d_stu/stu_tec_ours_centerpoint_pillar_005rate_voxel_tec/latest.pth

GPUS=8 GPUS_PER_NODE=8 CPUS_PER_TASK=16 sh ./tools/slurm_train.sh AD_GVT_A100_40G v_tec_3 \
    projects/configs/point3d_stu/stu_tec_ours_centerpoint_pillar_002rate_voxel_tec.py \
    /mnt/lustrenew/gaohongzhi/work_dirs/point_objDGCNN/point3d_stu/stu_tec_ours_centerpoint_pillar_002rate_voxel_tec/ \
    --resume-from /mnt/lustrenew/gaohongzhi/work_dirs/point_objDGCNN/point3d_stu/stu_tec_ours_centerpoint_pillar_002rate_voxel_tec/latest.pth
```

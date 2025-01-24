# --------------------------------------- centerpoint -------------------------------------------

## voxel
centerpoint_01voxel_second_secfpn_4x8_cyclic_20e_nus_sp_wovel_002rate
centerpoint_01voxel_second_secfpn_4x8_cyclic_20e_nus_sp_wovel_005rate
centerpoint_01voxel_second_secfpn_4x8_cyclic_20e_nus_sp_wovel_01rate
centerpoint_01voxel_second_secfpn_4x8_cyclic_20e_nus_sp_wovel_02rate
centerpoint_01voxel_second_secfpn_4x8_cyclic_20e_nus_sp_wovel_05rate
centerpoint_01voxel_second_secfpn_4x8_cyclic_20e_nus_sp_wovel
```bash
GPUS=8 GPUS_PER_NODE=8 CPUS_PER_TASK=32 sh ./tools/slurm_train.sh AD_GVT_A100_40G ctpt_stu_1 \
    projects/configs/wss_sp_ctpt/centerpoint_01voxel_second_secfpn_4x8_cyclic_20e_nus_sp_wovel_002rate.py \
    /mnt/lustrenew/gaohongzhi/work_dirs/point_objDGCNN/wss_sp_ctpt/centerpoint_01voxel_second_secfpn_4x8_cyclic_20e_nus_sp_wovel_002rate/

GPUS=8 GPUS_PER_NODE=8 CPUS_PER_TASK=32 sh ./tools/slurm_train.sh AD_GVT_A100_40G ctpt_stu_2 \
    projects/configs/wss_sp_ctpt/centerpoint_01voxel_second_secfpn_4x8_cyclic_20e_nus_sp_wovel_005rate.py \
    /mnt/lustrenew/gaohongzhi/work_dirs/point_objDGCNN/wss_sp_ctpt/centerpoint_01voxel_second_secfpn_4x8_cyclic_20e_nus_sp_wovel_005rate/

GPUS=8 GPUS_PER_NODE=8 CPUS_PER_TASK=32 sh ./tools/slurm_train.sh AD_GVT_A100_40G ctpt_stu_3 \
    projects/configs/wss_sp_ctpt/centerpoint_01voxel_second_secfpn_4x8_cyclic_20e_nus_sp_wovel_01rate.py \
    /mnt/lustrenew/gaohongzhi/work_dirs/point_objDGCNN/wss_sp_ctpt/centerpoint_01voxel_second_secfpn_4x8_cyclic_20e_nus_sp_wovel_01rate/

GPUS=8 GPUS_PER_NODE=8 CPUS_PER_TASK=32 sh ./tools/slurm_train.sh AD_GVT_A100_40G ctpt_stu_4 \
    projects/configs/wss_sp_ctpt/centerpoint_01voxel_second_secfpn_4x8_cyclic_20e_nus_sp_wovel_02rate.py \
    /mnt/lustrenew/gaohongzhi/work_dirs/point_objDGCNN/wss_sp_ctpt/centerpoint_01voxel_second_secfpn_4x8_cyclic_20e_nus_sp_wovel_02rate/

GPUS=8 GPUS_PER_NODE=8 CPUS_PER_TASK=32 sh ./tools/slurm_train.sh AD_GVT_A100_40G ctpt_stu_5 \
    projects/configs/wss_sp_ctpt/centerpoint_01voxel_second_secfpn_4x8_cyclic_20e_nus_sp_wovel_05rate.py \
    /mnt/lustrenew/gaohongzhi/work_dirs/point_objDGCNN/wss_sp_ctpt/centerpoint_01voxel_second_secfpn_4x8_cyclic_20e_nus_sp_wovel_05rate/

GPUS=8 GPUS_PER_NODE=8 CPUS_PER_TASK=32 sh ./tools/slurm_train.sh AD_GVT_A100_40G ctpt_stu_6 \
    projects/configs/wss_sp_ctpt/centerpoint_01voxel_second_secfpn_4x8_cyclic_20e_nus_sp_wovel.py \
    /mnt/lustrenew/gaohongzhi/work_dirs/point_objDGCNN/wss_sp_ctpt/centerpoint_01voxel_second_secfpn_4x8_cyclic_20e_nus_sp_wovel/
```

## pillar
centerpoint_02pillar_second_secfpn_4x8_cyclic_20e_nus_sp_wovel_002rate
centerpoint_02pillar_second_secfpn_4x8_cyclic_20e_nus_sp_wovel_005rate
centerpoint_02pillar_second_secfpn_4x8_cyclic_20e_nus_sp_wovel_01rate
centerpoint_02pillar_second_secfpn_4x8_cyclic_20e_nus_sp_wovel_02rate
centerpoint_02pillar_second_secfpn_4x8_cyclic_20e_nus_sp_wovel_05rate
centerpoint_02pillar_second_secfpn_4x8_cyclic_20e_nus_sp_wovel
```bash
GPUS=8 GPUS_PER_NODE=8 CPUS_PER_TASK=32 sh ./tools/slurm_train.sh AD_GVT_A100_40G ctpt1_stu_1 \
    projects/configs/wss_sp_ctpt/centerpoint_02pillar_second_secfpn_4x8_cyclic_20e_nus_sp_wovel_002rate.py \
    /mnt/lustrenew/gaohongzhi/work_dirs/point_objDGCNN/wss_sp_ctpt/centerpoint_02pillar_second_secfpn_4x8_cyclic_20e_nus_sp_wovel_002rate/

GPUS=8 GPUS_PER_NODE=8 CPUS_PER_TASK=32 sh ./tools/slurm_test.sh AD_GVT_A100_40G tec_test3 \
    projects/configs/wss_sp_ctpt/centerpoint_02pillar_second_secfpn_4x8_cyclic_20e_nus_sp_wovel_002rate.py \
    /mnt/lustrenew/gaohongzhi/work_dirs/point_objDGCNN/wss_sp_ctpt/centerpoint_02pillar_second_secfpn_4x8_cyclic_20e_nus_sp_wovel_002rate/latest.pth --eval bbox

GPUS=8 GPUS_PER_NODE=8 CPUS_PER_TASK=32 sh ./tools/slurm_train.sh AD_GVT_A100_40G ctpt1_stu_2 \
    projects/configs/wss_sp_ctpt/centerpoint_02pillar_second_secfpn_4x8_cyclic_20e_nus_sp_wovel_005rate.py \
    /mnt/lustrenew/gaohongzhi/work_dirs/point_objDGCNN/wss_sp_ctpt/centerpoint_02pillar_second_secfpn_4x8_cyclic_20e_nus_sp_wovel_005rate/ \
    --resume-from /mnt/lustrenew/gaohongzhi/work_dirs/point_objDGCNN/wss_sp_ctpt/centerpoint_02pillar_second_secfpn_4x8_cyclic_20e_nus_sp_wovel_005rate/latest.pth

GPUS=8 GPUS_PER_NODE=8 CPUS_PER_TASK=32 sh ./tools/slurm_train.sh AD_GVT_A100_40G ctpt1_stu_3 \
    projects/configs/wss_sp_ctpt/centerpoint_02pillar_second_secfpn_4x8_cyclic_20e_nus_sp_wovel_01rate.py \
    /mnt/lustrenew/gaohongzhi/work_dirs/point_objDGCNN/wss_sp_ctpt/centerpoint_02pillar_second_secfpn_4x8_cyclic_20e_nus_sp_wovel_01rate/

GPUS=8 GPUS_PER_NODE=8 CPUS_PER_TASK=32 sh ./tools/slurm_train.sh AD_GVT_A100_40G ctpt1_stu_4 \
    projects/configs/wss_sp_ctpt/centerpoint_02pillar_second_secfpn_4x8_cyclic_20e_nus_sp_wovel_02rate.py \
    /mnt/lustrenew/gaohongzhi/work_dirs/point_objDGCNN/wss_sp_ctpt/centerpoint_02pillar_second_secfpn_4x8_cyclic_20e_nus_sp_wovel_02rate/

GPUS=8 GPUS_PER_NODE=8 CPUS_PER_TASK=32 sh ./tools/slurm_train.sh AD_GVT_A100_40G ctpt1_stu_5 \
    projects/configs/wss_sp_ctpt/centerpoint_02pillar_second_secfpn_4x8_cyclic_20e_nus_sp_wovel_05rate.py \
    /mnt/lustrenew/gaohongzhi/work_dirs/point_objDGCNN/wss_sp_ctpt/centerpoint_02pillar_second_secfpn_4x8_cyclic_20e_nus_sp_wovel_05rate/

GPUS=8 GPUS_PER_NODE=8 CPUS_PER_TASK=32 sh ./tools/slurm_train.sh AD_GVT_A100_40G ctpt1_stu_6 \
    projects/configs/wss_sp_ctpt/centerpoint_02pillar_second_secfpn_4x8_cyclic_20e_nus_sp_wovel.py \
    /mnt/lustrenew/gaohongzhi/work_dirs/point_objDGCNN/wss_sp_ctpt/centerpoint_02pillar_second_secfpn_4x8_cyclic_20e_nus_sp_wovel/
```



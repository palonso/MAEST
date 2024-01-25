set -e

NCCL_SOCKET_IFNAME=vlan884 NCCL_IB_DISABLE=1 MASTER_ADDR=10.55.0.129 MASTER_PORT=6666 NODE_RANK=0 CUDA_VISIBLE_DEVICES=2,3 python ex_maest.py with maest_20s_from_passt_pretrain \
    trainer.num_sanity_val_steps=0 \
    trainer.num_nodes=1 \
    trainer.devices=2 \
    maest.n_classes=519 \
    datamodule.groundtruth_train=/home/palonso/reps/dt-training/src/preprocessing/231026-preprocessing/groundtruth_salmorejo.pk.fix \
    datamodule.groundtruth_val=/home/palonso/reps/dt-training/src/preprocessing/231026-preprocessing/groundtruth_val.pk.fix \
    datamodule.groundtruth_test=/home/palonso/reps/dt-training/src/preprocessing/231026-preprocessing/groundtruth_salmorejo.pk.fix \
    datamodule.base_dir=/data0/palonso/data/ \
    datamodule.base_dir_val=/home/palonso/data/discotube/discotube-specs/

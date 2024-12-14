set -e

# NCCL_SOCKET_IFNAME=vlan884 MASTER_ADDR=10.55.0.129 MASTER_PORT=6666 NCCL_IB_DISABLE=1 NODE_RANK=0 NCCL_BLOCKING_WAIT=0  

CUDA_VISIBLE_DEVICES=0,1,2,3 NCCL_BLOCKING_WAIT=0 python ex_maest.py -l INFO with maest_30s_from_passt_pretrain \
    ckpt_path="exp_logs/lightning_logs/428/checkpoints/epoch=121.no_swa.ckpt" \
    trainer.num_sanity_val_steps=0 \
    trainer.num_nodes=1 \
    trainer.devices=4 \
    trainer.log_every_n_steps=100 \
    datamodule.batch_size_train=6 \
    datamodule.batch_size_test=6 \
    maest.n_classes=519 \
    datamodule.groundtruth_train=merged.pk \
    datamodule.groundtruth_val=/home/palonso/reps/dt-training/src/preprocessing/231026-preprocessing/groundtruth_val.pk.fix \
    datamodule.groundtruth_test=/home/palonso/reps/dt-training/src/preprocessing/231026-preprocessing/groundtruth_salmorejo.pk.fix \
    datamodule.base_dir=/mnt/projects/discotube-melspectrograms/ \
    datamodule.base_dir_val=/home/palonso/data/discotube/discotube-specs/ \
    # ckpt_path="exp_logs/lightning_logs/306/checkpoints/epoch=44-val_loss=0.01-best.ckpt " \
    # ckpt_path="exp_logs/lightning_logs/306/checkpoints/epoch=57.no.swa.ckpt" \
    # ckpt_path="exp_logs/lightning_logs/306/checkpoints/epoch=57.no.swa.ckpt" \

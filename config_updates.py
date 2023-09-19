from sacred import Ingredient


def add_configs(ex):
    """
    This functions add generic configuration for the experiments, such as mix-up, architectures, etc...
    """

    @ex.named_config
    def mini_train():
        "limit training/validation to 5 batches for debbuging."
        trainer = dict(limit_train_batches=5, limit_val_batches=5)

        datamodule = {"groundtruth_val": "discogs/gt_val_all_400l_super_clean.pk"}

    #  Experiments from
    #  EFFICIENT SUPERVISED TRAINING OF AUDIO TRANSFORMERS FOR MUSIC REPRESENTATION LEARNING

    # Section 4.2. Impact of initial weights
    ########################################

    # Pretraining settings
    @ex.named_config
    def maest_10s_random_weights_pretrain():
        "time encodings for up to 10 seconds, and random initialization"

        datamodule = {"clip_length": 10}

        maest = {
            "arch": "passt_s_swa_p16_128_ap476",
            "pretrained": False,
            "input_t": 10 * 16000 // 256,
            "s_patchout_t": 30,
        }

    @ex.named_config
    def maest_10s_from_deit_pretrain():
        "time encodings for up to 10 seconds and initializaiton to the DeiT weights"

        datamodule = {"clip_length": 10}

        maest = {
            "arch": "passt_deit_bd_p16_384",
            "pretrained": True,
            "input_t": 10 * 16000 // 256,
            "s_patchout_t": 30,
        }

    @ex.named_config
    def maest_10s_from_passt_pretrain():
        "time encodings for up to 10 seconds and initializaiton to the PaSST weights"

        datamodule = {"clip_length": 10}

        maest = {
            "arch": "passt_s_swa_p16_128_ap476",
            "pretrained": True,
            "input_t": 10 * 16000 // 256,
            "s_patchout_t": 30,
        }

    # Inference settings

    @ex.named_config
    def maest_10s_random_weights_inference():
        "time encodings for up to 10 seconds, and random initialization (from scratch)"

        datamodule = {"clip_length": 10}

        maest = {
            "arch": "discogs-maest-10s-fs-129e",
            "pretrained": False,
            "input_t": 10 * 16000 // 256,
            "s_patchout_t": 30,
        }

        predict = {"transformer_block": 7}

    @ex.named_config
    def maest_10s_from_deit_inference():
        "time encodings for up to 10 seconds and initializaiton to the DeiT weights"

        datamodule = {"clip_length": 10}

        maest = {
            "arch": "discogs-maest-10s-dw-75e",
            "pretrained": True,
            "input_t": 10 * 16000 // 256,
            "s_patchout_t": 30,
        }

        predict = {"transformer_block": 7}

    @ex.named_config
    def maest_10s_from_passt_inference():
        "time encodings for up to 10 seconds and initializaiton to the PaSST weights"

        datamodule = {"clip_length": 10}

        maest = {
            "arch": "discogs-maest-10s-pw-129e",
            "pretrained": True,
            "input_t": 10 * 16000 // 256,
            "s_patchout_t": 30,
        }

        predict = {"transformer_block": 7}

    # Section 4.3. Effect of the input sequence length
    ##################################################

    # Pretraining settings

    @ex.named_config
    def passt_discogs_5sec_pretrain():
        "time encodings for up to 5 seconds"

        datamodule = {"clip_length": 5}

        maest = {
            "arch": "passt_s_swa_p16_128_ap476",
            "pretrained": True,
            "input_t": 5 * 16000 // 256,
            "s_patchout_t": 30,
        }

    @ex.named_config
    def passt_discogs_10sec_pretrain():
        "time encodings for up to 10 seconds"

        datamodule = {"clip_length": 10}

        maest = {
            "arch": "passt_s_swa_p16_128_ap476",
            "input_t": 10 * 16000 // 256,
            "s_patchout_t": 30,
        }

    @ex.named_config
    def passt_discogs_20sec_pretrain():
        "time encodings for up to 20 seconds"

        datamodule = {"clip_length": 20}

        maest = {
            "arch": "passt_s_swa_p16_128_ap476",
            "input_t": 20 * 16000 // 256,
            "s_patchout_t": 60,
        }

    @ex.named_config
    def passt_discogs_30sec_pretrain():
        "time encodings for up to 30 seconds"

        datamodule = {"clip_length": 30}

        maest = {
            "arch": "passt_s_swa_p16_128_ap476",
            "input_t": 30 * 16000 // 256,
            "s_patchout_t": 90,
        }

    # Inference settings

    @ex.named_config
    def passt_discogs_5sec_inference():
        "time encodings for up to 5 seconds"

        datamodule = {"clip_length": 5}

        maest = {
            "arch": "discogs-maest-5s-fs-129e",
            "pretrained": True,
            "input_t": 5 * 16000 // 256,
            "s_patchout_t": 30,
        }

        predict = {"transformer_block": 7}

    @ex.named_config
    def passt_discogs_10sec_inference():
        "time encodings for up to 10 seconds"

        datamodule = {"clip_length": 10}

        maest = {
            "arch": "discogs-maest-10s-fs-129e",
            "input_t": 10 * 16000 // 256,
            "s_patchout_t": 30,
        }

        predict = {"transformer_block": 7}

    @ex.named_config
    def passt_discogs_20sec_inference():
        "time encodings for up to 20 seconds"

        datamodule = {"clip_length": 20}

        maest = {
            "arch": "discogs-maest-20s-fs-129e",
            "input_t": 20 * 16000 // 256,
            "s_patchout_t": 60,
        }

        predict = {"transformer_block": 7}

    @ex.named_config
    def passt_discogs_30sec_inference():
        "time encodings for up to 30 seconds"

        datamodule = {"clip_length": 30}

        maest = {
            "arch": "discogs-maest-30s-fs-129e",
            "input_t": 30 * 16000 // 256,
            "s_patchout_t": 90,
        }

        predict = {"transformer_block": 7}

    # Teacher/student setup (unreleased experiment due to uncertain results).

    @ex.named_config
    def maest_30s_teacher_student_pretrain():
        "time encodings for up to 30 seconds"
        "using a teacher classifier"

        datamodule = {
            "batch_size_train": 4,
            "clip_length": 30,
            "teacher_student": {
                "do": True,
                "teacher_target_base_dir": "/home/palonso/reps/PaSST/logits/discogs/30sec/swa/11/",
            }
        }

        maest = {
            "arch": "passt_s_swa_p16_128_ap476",
            "input_t": 30 * 16000 // 256,
            "s_patchout_t": 90,
            "distilled_type": "separated",
        }

    # Downstream evaluation pipeline
    ################################

    # Embedding extraction

    @ex.named_config
    def target_mtt():
        "Target the MTT (MagnaTagATune dataset)"

        datamodule = {
            "groundtruth_predict": "datasets/mtt/groundtruth-all.pk",
            "base_dir": "datasets/mtt/data/mtt/melspec/",
        }

        predict = {
            "out_dir": "outputs/embeddings/mtt/",
        }

    # Training probes

    @ex.named_config
    def target_mtt_tl():
        "Target the MTT (MagnaTagATune dataset)"

        data = {
            "metadata_dir": "datasets/mtt/",
            "base_dir": "outputs/embeddings/mtt/10sec/swa/11/",
        }

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

    @ex.named_config
    def maest_10s_random_weights_pretrain():
        "time encodings for up to 10 seconds, and random initialization"

        datamodule = {"clip_length": 10}

        net = {
            "arch": "passt_s_swa_p16_128_ap476",
            "pretrained": False,
            "input_t": 10 * 16000 // 256,
            "s_patchout_t": 30,
        }

    @ex.named_config
    def maest_10s_from_deit_pretrain():
        "time encodings for up to 10 seconds and initializaiton to the DeiT weights"

        datamodule = {"clip_length": 10}

        net = {
            "arch": "passt_deit_bd_p16_384",
            "pretrained": True,
            "input_t": 10 * 16000 // 256,
            "s_patchout_t": 30,
        }

    @ex.named_config
    def maest_10s_from_passt_pretrain():
        "time encodings for up to 10 seconds and initializaiton to the PaSST weights"

        datamodule = {"clip_length": 10}

        net = {
            "arch": "passt_s_swa_p16_128_ap476",
            "pretrained": True,
            "input_t": 10 * 16000 // 256,
            "s_patchout_t": 30,
        }

    @ex.named_config
    def maest_10s_random_weights_inference():
        "time encodings for up to 10 seconds, and random initialization (from scratch)"

        datamodule = {"clip_length": 10}

        net = {
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

        net = {
            "arch": "discogs-maest-10s-dw-75ee",
            "pretrained": True,
            "input_t": 10 * 16000 // 256,
            "s_patchout_t": 30,
        }

        predict = {"transformer_block": 7}

    @ex.named_config
    def maest_10s_from_passt_inference():
        "time encodings for up to 10 seconds and initializaiton to the PaSST weights"

        datamodule = {"clip_length": 10}

        net = {
            "arch": "discogs-maest-10s-pw-129e",
            "pretrained": True,
            "input_t": 10 * 16000 // 256,
            "s_patchout_t": 30,
        }

        predict = {"transformer_block": 7}

    # Section 4.3. Effect of the input sequence length
    ##################################################

    @ex.named_config
    def passt_discogs_5sec():
        "time encodings for up to 5 seconds"

        datamodule = {"clip_length": 5}

        net = {
            "arch": "passt_s_swa_p16_128_ap476",
            "pretrained": True,
            "input_t": 5 * 16000 // 256,
            "s_patchout_t": 30,
        }

    @ex.named_config
    def passt_discogs_10sec():
        "time encodings for up to 10 seconds"

        datamodule = {"clip_length": 10}

        net = {
            "arch": "passt_s_swa_p16_128_ap476_discogs",
            "input_t": 10 * 16000 // 256,
            "s_patchout_t": 30,
        }

    @ex.named_config
    def passt_discogs_20sec():
        "time encodings for up to 20 seconds"

        datamodule = {"clip_length": 20}

        net = {
            "arch": "passt_s_swa_p16_128_ap476_discogs",
            "input_t": 20 * 16000 // 256,
            "s_patchout_t": 60,
        }

    @ex.named_config
    def passt_discogs_30sec():
        "time encodings for up to 30 seconds"

        datamodule = {"clip_length": 30}

        net = {
            "arch": "passt_s_swa_p16_128_ap476_discogs",
            "input_t": 30 * 16000 // 256,
            "s_patchout_t": 90,
        }

    @ex.named_config
    def passt_discogs_10sec_fe():
        "use PaSST model pretrained on Audioset (with SWA) ap=476; time encodings for up to 10 seconds with frequency-wise embeddings"
        # python ex_audioset.py evaluate_only with passt_s_ap476
        models = {
            "net": Ingredient(
                "models.passt.model_ing",
                input_tdim=625,
                s_patchout_t=30,
                n_patches_t=87,
                embed="freq_embed",
                arch="passt_s_swa_p16_128_ap476",
            )
        }
        basedataset = dict(clip_length=10)

    @ex.named_config
    def teacher_target():
        "using a teacher classifier"
        # python ex_audioset.py evaluate_only with passt_s_ap476
        models = {
            "net": Ingredient(
                "models.passt.model_ing",
                distilled_type="separated",
                s_patchout_t=90,
            )
        }
        basedataset = dict(
            teacher_target=True,
            teacher_target_base_dir="logits/discogs/30sec/swa/11/",
            teacher_target_threshold=0.45,
        )

    @ex.named_config
    def passt_discogs5sec_inference():
        "extracting embeddings with the 5secs model"

        inference = dict(
            n_block=11,
        )

        models = {
            "net": Ingredient(
                "models.passt.model_ing",
                arch="passt_s_swa_p16_128_ap476_discogs",
                checkpoint="output/discogs/230403-122647/4b37a9b508d244978f05b6feba6da75e_0/checkpoints/epoch=129-step=541709.ckpt",
                use_swa=True,
            )
        }

    @ex.named_config
    def passt_discogs10sec_inference():
        "extract embeddings 10secs"

        inference = dict(
            n_block=11,
        )

        models = {
            "net": Ingredient(
                "models.passt.model_ing",
                arch="passt_s_swa_p16_128_ap476_discogs",
                checkpoint="output/discogs/945693489efb439996d141b35a2ec63d/checkpoints/epoch=129-step=541709.ckpt",
                use_swa=True,
            )
        }

    @ex.named_config
    def passt_discogs20sec_inference():
        "extract embeddings 20secs"

        inference = dict(
            n_block=11,
        )

        models = {
            "net": Ingredient(
                "models.passt.model_ing",
                arch="passt_s_swa_p16_128_ap476_discogs",
                checkpoint="output/discogs/230323-020758/discogs/d42e938b9d2d4d389c555c2f508fe277/checkpoints/epoch=129-step=541709.ckpt",
                use_swa=True,
            )
        }

    @ex.named_config
    def passt_discogs30sec_inference():
        "extract embeddings 30s"

        inference = dict(
            n_block=11,
        )

        models = {
            "net": Ingredient(
                "models.passt.model_ing",
                arch="passt_s_swa_p16_128_ap476_discogs",
                # checkpoint="output/discogs/230325-141455/discogs/ec14e4ef8b104814aa5ec985803bb9d8/checkpoints/epoch=46-step=195848.ckpt",
                # checkpoint="output/discogs/230325-141455/discogs/ec14e4ef8b104814aa5ec985803bb9d8/checkpoints/epoch=66-step=279188.ckpt",
                # checkpoint="output/discogs/230325-141455/discogs/ec14e4ef8b104814aa5ec985803bb9d8/checkpoints/epoch=90-step=379196.ckpt",
                checkpoint="output/discogs/230325-141455/discogs/ec14e4ef8b104814aa5ec985803bb9d8/checkpoints/epoch=129-step=541709.ckpt",
                use_swa=True,
            )
        }

    @ex.named_config
    def dynamic_roll():
        # dynamically roll the spectrograms/waveforms
        # updates the dataset config
        basedataset = dict(roll=True, roll_conf=dict(axis=1, shift_range=10000))

    # extra commands

    @ex.command
    def test_loaders_train_speed():
        # test how fast data is being loaded from the data loaders.
        itr = ex.datasets.training.get_iter()
        import time

        start = time.time()
        print("hello")
        for i, b in enumerate(itr):
            if i % 20 == 0:
                print(f"{i}/{len(itr)}", end="\r")
        end = time.time()
        print("totoal time:", end - start)
        start = time.time()
        print("retry:")
        for i, b in enumerate(itr):
            if i % 20 == 0:
                print(f"{i}/{len(itr)}", end="\r")
        end = time.time()
        print("totoal time:", end - start)

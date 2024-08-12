import torch
from models.maest import maest
from transformers import ASTConfig, ASTForAudioClassification
from unicodedata import normalize

from models.discogs_labels import discogs_400labels, discogs_519labels
from maest.feature_extraction_maest import MAESTFeatureExtractor

org = "mtg-upf"

models = [
    # "discogs-maest-10s-fs-129e-swa",
    # "discogs-maest-10s-pw-129e-swa",
    # "discogs-maest-10s-dw-75e-swa",
    # "discogs-maest-5s-pw-129e-swa",
    # "discogs-maest-20s-pw-129e-swa",
    # "discogs-maest-30s-pw-129e-swa",
    # "discogs-maest-30s-pw-73e-ts-swa",
    # "discogs-maest-10s-fs-129e",
    # "discogs-maest-10s-pw-129e",
    # "discogs-maest-10s-dw-75e",
    # "discogs-maest-5s-pw-129e",
    # "discogs-maest-20s-pw-129e",
    # "discogs-maest-30s-pw-129e",
    # "discogs-maest-30s-pw-73e-ts",
    "discogs-maest-30s-pw-129e-519l",
]


def update_state_dict_names(state_dict):
    replacements = [
        ("blocks.", "audio_spectrogram_transformer.encoder.layer."),
        ("cls_token", "audio_spectrogram_transformer.embeddings.cls_token"),
        ("dist_token", "audio_spectrogram_transformer.embeddings.distillation_token"),
        (
            "patch_embed.proj.",
            "audio_spectrogram_transformer.embeddings.patch_embeddings.projection.",
        ),
        ("norm.", "audio_spectrogram_transformer.layernorm."),
        # domain specific
        ("norm1.", "layernorm_before."),
        ("norm2.", "layernorm_after."),
        ("mlp.fc1.", "intermediate.dense."),
        ("mlp.fc2.", "output.dense."),
        ("attn.proj.", "attention.output.dense."),
        # fix head
        ("head.0.", "classifier.layernorm."),
        ("head.1.", "classifier.dense."),
    ]

    for inp, out in replacements:
        state_dict = {k.replace(inp, out): v for k, v in state_dict.items()}

    return state_dict


def split_qkv_matrices(state_dict):
    keys_remove = []
    new_values = dict()
    for k, v in state_dict.items():
        if "qkv" in k:
            keys_remove.append(k)
            qu, ke, va = v.chunk(3, dim=0)
            _, _, _, layer, _, _, type = k.split(".")
            for mat, name in zip([qu, ke, va], ["query", "key", "value"]):
                new_values[
                    f"audio_spectrogram_transformer.encoder.layer.{layer}.attention.attention.{name}.{type}"
                ] = mat

    state_dict.update(new_values)

    for k in keys_remove:
        del state_dict[k]

    return state_dict


def recombine_pos_embeddings(state_dict):
    pos_embs = state_dict["freq_new_pos_embed"] + state_dict["time_new_pos_embed"]
    print(state_dict["time_new_pos_embed"].shape)
    # Flatten time/freq pos embeddings
    pos_embs = torch.flatten(pos_embs, 2, 3)
    pos_embs = torch.transpose(pos_embs, 1, 2)

    # print(state_dict["new_pos_embed"].shape)
    # print(pos_embs.shape)
    pos_embs = torch.cat((state_dict["new_pos_embed"], pos_embs), axis=1)

    state_dict[
        "audio_spectrogram_transformer.embeddings.position_embeddings"
    ] = pos_embs

    del state_dict["freq_new_pos_embed"]
    del state_dict["time_new_pos_embed"]
    del state_dict["new_pos_embed"]

    return state_dict


def get_max_length(maest_model):
    if "5s" in maest_model:
        return 316
    elif "10s" in maest_model:
        return 626
    elif "20s" in maest_model:
        return 1256
    elif "30s" in maest_model:
        return 1876


def remove_dist_head(atate_dict):
    del state_dict["head_dist.weight"]
    del state_dict["head_dist.bias"]

    return state_dict


# print("QKV values from AST")
# print("Query: ", model_ast.encoder.layer[0].attention.attention.query.weight.shape)
# print("Key: ", model_ast.encoder.layer[0].attention.attention.key.weight.shape)
# print("Value: ", model_ast.encoder.layer[0].attention.attention.value.weight.shape)
# print("KQV values from Maest", state_dict["encoder.layer.8.attn.qkv.weight"].shape)


MAESTFeatureExtractor.register_for_auto_class("AutoFeatureExtractor")

for model in models:
    if "519l" in model:
        labels = discogs_519labels
    else:
        labels = discogs_400labels

    n_classes = len(labels)

    max_length = get_max_length(model)
    configuration = ASTConfig(
        num_mel_bins=96,
        max_length=max_length,
        time_stride=10,
        id2label={k: normalize("NFKC", v) for k, v in enumerate(labels)},
        label2id={normalize("NFKC", v): k for k, v in enumerate(labels)},
        layer_norm_eps=1e-6,
        torch_dtype="float16",
    )
    model_ast = ASTForAudioClassification(configuration)

    print(f"Uploading {model} to {org}")
    maest_feature_extractor = MAESTFeatureExtractor(max_length=max_length)

    model_maest = maest(model, n_classes=n_classes)
    state_dict = update_state_dict_names(model_maest.state_dict())
    state_dict = split_qkv_matrices(state_dict)
    state_dict = recombine_pos_embeddings(state_dict)
    state_dict = remove_dist_head(state_dict)

    model_ast.load_state_dict(state_dict)

    model_ast.push_to_hub(f"{org}/{model}")
    maest_feature_extractor.push_to_hub(f"{org}/{model}")

print("finish!")

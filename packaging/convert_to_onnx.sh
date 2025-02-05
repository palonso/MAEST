set -e

for model in \
    discogs-maest-30s-pw-129e-519l \
    # discogs-maest-10s-fs-129e \
    # discogs-maest-10s-dw-75e \
    # discogs-maest-5s-pw-129e \
    # discogs-maest-10s-pw-129e \
    # discogs-maest-20s-pw-129e \
    # discogs-maest-30s-pw-129e \
    # discogs-maest-30s-pw-73e-ts
do
    echo processing "$model"

    git clone git@hf.co:mtg-upf/"$model" input_models/

    cd input_models/"$model"

    # Set and env var with the HF_API_TOKEN (https://huggingface.co/settings/tokens)
    git remote set-url origin https://palonso:{$HF_API_TOKEN}@huggingface.co/mtg-upf/"$model"

    git lfs pull

    cd ../../

    python safetensors_to_pytorch.py --base_path input_models/"$model"


    optimum-cli export onnx \
        --model input_models/"$model" \
        --task audio-classification \
        output_models/ 

    mv output_models/model.onnx output_models/"$model".onnx
done

set -e
dir="/home/palonso/reps/keras-onnx-tensorflow-converter/src/"

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
    cp output_models/${model}.onnx output_models/${model}.onnx.tmp

    echo "Fixing model name"
    python ${dir}change_interface_names.py \
	-f \
        output_models/${model}.onnx.tmp \
        output_models/${model}.onnx.tmp \
	-i melspectrogram \
	-o logits

    echo "Adding sigmoid node"
    # First we need to add a Sigmoid output node since it was not included in the ONNX model.
    python ${dir}add_output_node.py \
        output_models/${model}.onnx.tmp \
        output_models/${model}.onnx.tmp \
	/classifier/dense/Gemm \
        activations \
        --node-type Sigmoid \
        --output-shape batch_size,519 \

    echo "Adding output nodes"
    for layer in {0..11}
    do
        python ${dir}add_output_node.py \
            output_models/${model}.onnx.tmp \
            output_models/${model}.onnx.tmp \
            /audio_spectrogram_transformer/encoder/layer.${layer}/output/Add \
            layer_${layer}_tokens \
            --output-shape batch_size,n,768
    done

    echo "Converting to SavedModel"
    python ${dir}o2sm.py -f output_models/${model}.onnx.tmp output_models/${model}.tmp

    echo "Converting to ProtoBuff"
    python ${dir}sm2pb.py -f output_models/${model}.tmp output_models/${model}.pb

    rm -r output_models/${model}.onnx.tmp

    echo "Done with ${model}"
done

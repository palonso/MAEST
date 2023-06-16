

# create conda env and activate
    conda create -n PMSST python=3.10
    conda activate PMSST

# install torch 2.0
    pip install numpy --pre torch torchvision torchaudio --force-reinstall --index-url https://download.pytorch.org/whl/nightly/cu118

# install rest of stuff
    pip install -r requirements.txt

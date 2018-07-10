#!/bin/bash

# parameters you need to specify:
# required: src, tar, src_dev, tar_dev, save_reader, train_data, dev_data
# optional: prependtitle, prependlinum, use_ir_pred, n_sentences
if ! [ -z ${prependtitle} ]; then
    prependtitle="--prependtitle"
fi

if [ ! -z ${prependlinum} ]; then
    prependlinum="--prependlinum"
fi

if [ ! -z ${use_ir_pred}]; then
    use_ir_pred="--use_ir_pred"
fi

if [ -z ${use_ir_pred} ]; then
    n_sentences=""
fi

pushd .
cd ~/fever

conda deactivate && conda deactivate > /dev/null
conda activate > /dev/null

python3 converter.py ${src} ${tar} ${prependtitle} ${prependlinum} ${use_ir_pred} ${n_sentences}
python3 converter.py ${src_dev} ${tar_dev} ${prependtitle} ${prependlinum} ${use_ir_pred} ${n_sentences}


cd ~/jack

conda deactivate && conda deactivate > /dev/null
conda activate && conda activate jack > /dev/null

PYTHONPATH=. anaconda-python3-gpu bin/jack-train.py with config='./conf/nli/fever/esim.yaml' save_dir=${save_reader} train=${tar} dev=${tar_dev} test=${tar_dev}


#!/usr/bin/env bash

# get the absolute path of this file
# (** This does not expand symlink)
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

pushd . > /dev/null
# clone takuma-ynd/jack.git
cd ${CURRENT_DIR}/../
git clone https://github.com/takuma-ynd/jack.git

# install requirements and Glove data
cd jack
python -m pip install -e .[tf]
# temporary disabled
bash ./data/GloVe/download.sh

popd > /dev/null

pushd . > /dev/null
cd ${CURRENT_DIR}/../
git clone https://github.com/takuma-ynd/fever-baselines.git
cd fever-baselines
pip install -r requirements.txt
popd > /dev/null

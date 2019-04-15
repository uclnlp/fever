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
bash ./data/GloVe/download.sh

popd > /dev/null

pushd . > /dev/null
cd ${CURRENT_DIR}/../
git clone https://github.com/takuma-ynd/fever-baselines.git
cd fever-baselines
pip install -r requirements.txt
popd > /dev/null

mkdir data
mkdir results

# download trained model
wget "http://tti-coin.jp/data/yoneda/fever/base+sampling2+evscores+rerank+train+dev+test-shared_test.ver0727_newaggr_submission.zip" -O "/tmp/base+sampling2+evscores+rerank+train+dev+test-shared_test.ver0727_newaggr_submission.zip"

unzip /tmp/base+sampling2+evscores+rerank+train+dev+test-shared_test.ver0727_newaggr_submission.zip -d /hexaf/fever/results

wget "http://tti-coin.jp/data/yoneda/fever/data.zip" -O "/tmp/data.zip"
unzip /tmp/data.zip -d /hexaf/fever

bash setup.sh

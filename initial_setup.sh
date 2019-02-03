#!/usr/bin/env bash

# get the absolute path of this file
# (** This does not expand symlink)
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

pushd . > /dev/null
# clone takuma-ynd/jack.git
cd ${THIS_FILE_PATH}/../
git clone https://github.com/takuma-ynd/jack.git
cd jack

# This part can fail depending on the python environment. -> better to manually run this.
# # if "python3" command is available, use that.
# if command -v python3 &>/dev/null; then
#     python3 -m pip install -e .[tf]
# else
#     python -m pip install -e .[tf]
# fi
# bash ./data/GloVe/download.sh
popd > /dev/null

mkdir data
mkdir results

download_if_not_exists() {
    if [ ! -f $2 ]; then
        wget $1 -O $2
    else
        echo "$2 already exists. skipping..."
    fi
}
download_if_not_exists "https://s3-eu-west-1.amazonaws.com/fever.public/train.jsonl" "data/train.jsonl"
download_if_not_exists "https://s3-eu-west-1.amazonaws.com/fever.public/shared_task_dev.jsonl" "data/dev.jsonl"
download_if_not_exists "https://s3-eu-west-1.amazonaws.com/fever.public/shared_task_test.jsonl" "data/test.jsonl"
download_if_not_exists "https://s3-eu-west-1.amazonaws.com/fever.public/wiki-pages.zip" "/tmp/wiki-pages.zip"
# download trained model
download_if_not_exists "http://tti-coin.jp/data/yoneda/fever/base+sampling2+evscores+rerank+train+dev+test-shared_test.ver0727_newaggr_submission.zip" "/tmp/base+sampling2+evscores+rerank+train+dev+test-shared_test.ver0727_newaggr_submission.zip"

if [ ! -d data/wiki-pages/wiki-pages ]; then
    mkdir data/wiki-pages
    unzip /tmp/wiki-pages.zip -d data/wiki-pages
    rm /tmp/wiki-pages.zip
fi

if [ ! -d results/base+sampling2+evscores+rerank+train+dev+test-shared_test.ver0727_newaggr_submission ]; then
    unzip /tmp/base+sampling2+evscores+rerank+train+dev+test-shared_test.ver0727_newaggr_submission.zip -d results
fi

bash setup.sh

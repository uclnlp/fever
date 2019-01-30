#!/usr/bin/env bash

# get the absolute path of this file
# (** This does not expand symlink)
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# clone takuma-ynd/jack.git
cd ${THIS_FILE_PATH}/../
git clone https://github.com/takuma-ynd/jack.git
cd jack

pushd . > /dev/null
# if "python3" command is available, use that.
if command -v python3 &>/dev/null; then
    python3 -m pip install -e .[tf]
else
    python -m pip install -e .[tf]
fi
bash ./data/GloVe/download.sh
popd > /dev/null

# mkdir data

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

if [ ! -d data/wiki-pages/wiki-pages ]; then
    mkdir data/wiki-pages
    unzip /tmp/wiki-pages.zip -d data/wiki-pages/wiki-pages
fi

bash setup.sh



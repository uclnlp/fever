#!/usr/bin/env bash

# super complicated but this just gets the current directory.
pushd . > /dev/null
THIS_FILE_PATH="${BASH_SOURCE[0]}"
if ([ -h "${THIS_FILE_PATH}" ]); then
    while([ -h "${THIS_FILE_PATH}" ]); do cd `dirname "$SCRIPT_PATH"`;
                                       THIS_FILE_PATH=`readlink "${SCRIPT_PATH}"`; done
fi
cd `dirname ${THIS_FILE_PATH}` > /dev/null
THIS_FILE_PATH=`pwd`;
popd  > /dev/null


pushd . > /dev/null
# clone takuma-ynd/jack.git
cd ${THIS_FILE_PATH}/../
git clone https://github.com/takuma-ynd/jack.git
cd jack

# if "python3" command is available, use that.
if command -v python3 &>/dev/null; then
    python3 -m pip install -e .[tf]
else
    python -m pip install -e .[tf]
fi
bash ./data/GloVe/download.sh
popd > /dev/null

mkdir data
wget https://s3-eu-west-1.amazonaws.com/fever.public/train.jsonl -O data/train.jsonl
wget https://s3-eu-west-1.amazonaws.com/fever.public/shared_task_dev.jsonl -O data/dev.jsonl
wget https://s3-eu-west-1.amazonaws.com/fever.public/shared_task_test.jsonl -O data/test.jsonl
wget https://s3-eu-west-1.amazonaws.com/fever.public/wiki-pages.zip -O /tmp/wiki-pages.zip
mkdir data/wiki-pages
unzip /tmp/wiki-pages.zip -d data/wiki-pages/wiki-pages

bash setup.sh



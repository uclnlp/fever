#!/usr/bin/env bash

# get the absolute path of this file
# (** This does not expand symlink)
PARENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "replacing **BASE_DIR** with ${PARENT_DIR}"

# escape slashes
ESCAPED_PARENT_DIR=$(echo ${PARENT_DIR}| sed 's/\//\\\//g')

CONFIG1=${PARENT_DIR}/fever/configs/submission_config.json
CONFIG2=${PARENT_DIR}/fever/configs/base_config.json
PYFILE=${PARENT_DIR}/fever/pipeline.py
MODEL1=${PARENT_DIR}/fever/results/base+sampling2+evscores+rerank+train+dev+test-shared_test.ver0727_newaggr_submission/reader/checkpoint
MODEL2=${PARENT_DIR}/fever/results/base+sampling2+evscores+rerank+train+dev+test-shared_test.ver0727_newaggr_submission/reader/shared_resources/config.yaml

# replace
sed -i -e "s/\*\*BASE_DIR\*\*/${ESCAPED_PARENT_DIR}/g" ${CONFIG1} ${CONFIG2} ${PYFILE} ${MODEL1} ${MODEL2}




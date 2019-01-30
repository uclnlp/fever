#!/usr/bin/env bash

# get the absolute path of this file
# (** This does not expand symlink)
PARENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "replacing **BASE_DIR** with ${PARENT_DIR}"

# escape slashes
ESCAPED_PARENT_DIR=$(echo ${PARENT_DIR}| sed 's/\//\\\//g')

# replace
sed -i -e "s/\*\*BASE_DIR\*\*/${ESCAPED_PARENT_DIR}/g" ${PARENT_DIR}/fever/configs/submission_config.json ${PARENT_DIR}/fever/configs/base_config.json ${PARENT_DIR}/fever/pipeline.py




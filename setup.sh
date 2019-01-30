#!/usr/bin/env bash

# get the absolute path of this file
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "replacing **BASE_DIR** with ${CURRENT_DIR}"

# escape slashes
ESCAPED_CURRENT_DIR=$(echo ${CURRENT_DIR}| sed 's/\//\\\//g')

# replace
sed -i -e "s/\*\*BASE_DIR\*\*/${ESCAPED_CURRENT_DIR}/g" ${CURRENT_DIR}/configs/submission_config.json ${CURRENT_DIR}/configs/base_config.json ${CURRENT_DIR}/pipeline.py




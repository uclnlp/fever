#!/usr/bin/env bash

# super complicated but this just gets the current directory.
pushd . > /dev/null
CURRENT_DIR="${BASH_SOURCE[0]}"
if ([ -h "${CURRENT_DIR}" ]); then
    while([ -h "${CURRENT_DIR}" ]); do cd `dirname "$SCRIPT_PATH"`;
                                       CURRENT_DIR=`readlink "${SCRIPT_PATH}"`; done
fi
cd `dirname ${CURRENT_DIR}` > /dev/null
CURRENT_DIR=`pwd`;
popd  > /dev/null
echo "replacing **BASE_DIR** with ${CURRENT_DIR}"

# escape slashes
ESCAPED_CURRENT_DIR=$(echo ${CURRENT_DIR}| sed 's/\//\\\//g')

# replace
sed -i "s/**BASE_DIR**/${ESCAPED_CURRENT_DIR}/g" ${CURRENT_DIR}/configs/submission_config.json ${CURRENT_DIR}/configs/base_config.json ${CURRENT_DIR}/pipeline.py




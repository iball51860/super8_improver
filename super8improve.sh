#!/usr/bin/env bash

COMMAND=${1}
PROJECT_PATH=${2}

docker run \
    -v ${PROJECT_PATH}:${PROJECT_PATH} \
    super8improver \
    python ./main.py ${COMMAND} ${PROJECT_PATH}
#!/bin/sh

docker run \
    -it \
    --rm \
    -p 8888:8888 \
    -v `pwd`:/home/jovyan/work \
    --name cython-tutorial \
    jupyter/scipy-notebook

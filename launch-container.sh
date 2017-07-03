#!/bin/sh

docker run \
    -it \
    --rm \
    -p 8888:8888 \
    -v `pwd`:/home/jovyan/ \
    --name cython-tutorial \
    jupyter/scipy-notebook:8e15d329f1e9

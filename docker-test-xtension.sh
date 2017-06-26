#!/bin/sh

docker exec \
    -it \
    cython-tutorial \
    /bin/sh -c "./test-xtension.sh"

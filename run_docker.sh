#!/bin/bash
set -e
set -x

SCRIPT_DIR=$(dirname $(realpath $0))
docker run -d --rm \
	--name zheng-flint \
	--privileged \
	--gpus all \
	--ipc=host \
	--ulimit memlock=-1 \
	--ulimit stack=6710886 \
	-v $SCRIPT_DIR:/workspace/chakra_fx \
	zheng-flint:1007 \
	tail -f /dev/null

docker exec -it zheng-flint /bin/bash
#!/bin/bash
set -e
set -x

SCRIPT_DIR=$(dirname $(realpath $0))
docker run -d --rm \
	--name chakra_fx \
	--privileged \
	--gpus all \
	--ipc=host \
	--ulimit memlock=-1 \
	--ulimit stack=6710886 \
	-v $SCRIPT_DIR:/workspace/chakra_fx \
	zheng/chakra_fx:0817 \
	tail -f /dev/null

docker exec -it chakra_fx /bin/bash

#!/bin/bash
set -e
set -x

SCRIPT_DIR=$(dirname $(realpath $0))
docker run -d --rm \
	--name flint \
	--privileged \
	--gpus all \
	--ipc=host \
	--ulimit memlock=-1 \
	--ulimit stack=6710886 \
	-v $SCRIPT_DIR:/workspace/chakra_fx \
	jyoo332/flint:latest \
	tail -f /dev/null

docker exec -it flint /bin/bash

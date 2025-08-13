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
	jyoo332/chakra_fx:nomount \
	tail -f /dev/null

docker exec -it chakra_fx /bin/bash

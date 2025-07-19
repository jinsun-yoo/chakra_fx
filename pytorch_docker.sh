docker run -d --rm \
	--name fxgraph_ASK_JINSUN1 \
	--privileged \
	--gpus all \
	--ipc=host \
	--ulimit memlock=-1 \
	--ulimit stack=6710886 \
	-v /nethome/jyoo332/chakra_fx:/workspace/chakra_fx \
	-v /nethome/jyoo332/.ssh:/workspace/.ssh \
	-v /nethome/jyoo332/pytorch:/workspace/pytorch \
	fxgraph_python:latest \
	tail -f /dev/null

	#--user $(id -u):$(id -g) \
docker exec -it fxgraph_ASK_JINSUN1 /bin/bash

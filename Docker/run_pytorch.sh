WORKSPACE=/home/dh/Workspace

# xhost +local:docker
docker run -it --rm \
	--gpus all \
	--net host \
	--shm-size 8G \
    -w $WORKSPACE \
	-v $WORKSPACE:$WORKSPACE \
	pytorch_1.12.1

# -v $HOME/.Xauthority:/root/.Xauthority:rw \
# -v /tmp/.X11-unix:/tmp/.X11-unix \
# -e DISPLAY=unix$DISPLAY \
# -e QT_X11_NO_MITSHM=1 \
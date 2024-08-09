# Use the container

```bash
$ DOCKER_IMG="soliderreid:dev" && docker build -f docker/Dockerfile -t $DOCKER_IMG --build-arg BASE_IMAGE=nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04 .

$ docker run --rm -it --entrypoint /bin/bash \
    --name soliderreid \
    --gpus all \
    --shm-size 96G \
    -v /home/$USER/workspace:/home/workspace \
    soliderreid:dev
```

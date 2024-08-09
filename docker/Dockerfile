ARG BASE_IMAGE=nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && apt-get install -y \
	ca-certificates python3-dev python3-pip python3-tk git wget ninja-build libgl1 libgl1-mesa-glx libglib2.0-0

RUN pip3 install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
COPY docker/requirements.txt /opt/tmp/requirements.txt
RUN pip3 install -r /opt/tmp/requirements.txt

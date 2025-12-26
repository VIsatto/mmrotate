ARG PYTORCH="2.0.1"
ARG CUDA="11.8"

FROM anibali/pytorch:${PYTORCH}-cuda${CUDA}-ubuntu22.04

RUN sudo apt-get update &&\
    sudo apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 libc6 &&\
    sudo rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir -U openmim
RUN mim install --no-cache-dir mmengine "mmcv>=2.0.0rc4,<2.1.0" "mmdet>=3.0.0rc2,<3.1.0"
RUN pip install numpy<2
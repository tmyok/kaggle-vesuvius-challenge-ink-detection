#!/bin/sh
# -*- coding: utf-8 -*-

docker run \
    -it \
    --rm \
    --gpus all \
    --shm-size=32g \
    --name kaggle_VCID \
    --volume $(pwd)/:/home/work/ \
    --volume ~/dataset/ink/input/:/home/work/input:ro \
    --volume ~/dataset/ink/output/:/home/work/output \
    --workdir /home/work/ \
    tmyok/pytorch:2.0.1-cuda11.8.0-cudnn8-tensorrt861-opencv470-ubuntu22.04
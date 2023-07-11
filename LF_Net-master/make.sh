#!/usr/bin/env bash
cd ./utils/

CUDA_PATH=/usr/local/cuda/
#CUDA_PATH=/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA

python build.py build_ext --inplace

cd ..

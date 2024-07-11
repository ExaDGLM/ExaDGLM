#!/bin/sh

CUDA_DIR="/opt/nvidia/hpc_sdk/Linux_x86_64/24.1/cuda/12.3/targets/x86_64-linux"

echo "utils/build/check_gpu_spec_nvidia"
nvcc utils/src/cuda/check_gpu_spec_nvidia.cu -o utils/build/check_gpu_spec_nvidia

echo "utils/src/python/gpu_info.so"
g++ -std=c++11 -shared -fPIC `python3 -m pybind11 --includes` utils/src/cuda/gpu_info.cpp -I/$CUDA_DIR/include -L/$CUDA_DIR/lib -lcudart -o utils/src/python/gpu_info.so

echo "dim3/src/python/uniform_3d_single_cpp.so"
g++ -O3 -Wall -shared -std=c++17 -fPIC `python3 -m pybind11 --includes` dim3/src/cpp/uniform_3d_single_core.cpp -o dim3/src/python/uniform_3d_single_cpp.so

echo "dim3/src/python/mesh_3d_cpp.so"
g++ -O3 -Wall -shared -std=c++17 -fPIC `python3 -m pybind11 --includes` dim3/src/cpp/mesh_3d_core.cpp -o dim3/src/python/mesh_3d_cpp.so

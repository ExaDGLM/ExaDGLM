#include <cuda_runtime.h>
#include <iostream>

int main() {
    cudaDeviceProp prop;
    int count;

    cudaGetDeviceCount(&count);

    for (int i = 0; i < count; i++) {
        cudaGetDeviceProperties(&prop, i);

        std::cout << "Device " << i << ":\n";
        std::cout << "Name: " << prop.name << "\n";
        std::cout << "Compute capability: " << prop.major << "." << prop.minor << "\n";
        std::cout << "Number of SM: " << prop.multiProcessorCount << "\n";
        std::cout << "Threads per warp: " << prop.warpSize << "\n";
        std::cout << "Max active warps per SM: " << prop.maxThreadsPerMultiProcessor / prop.warpSize << "\n";
        std::cout << "Max shared memory per SM: " << prop.sharedMemPerMultiprocessor << " bytes\n";
        std::cout << "Max shared memory per block: " << prop.sharedMemPerBlock << " bytes\n";
        std::cout << "Max constant memory: " << prop.totalConstMem << " bytes\n\n";
    }

    return 0;
}

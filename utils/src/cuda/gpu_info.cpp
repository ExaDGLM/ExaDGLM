#include <pybind11/pybind11.h>
#include <cuda_runtime.h>

namespace py = pybind11;

py::dict get_gpu_info() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);  // Assuming device 0

    py::dict gpu_info;
    gpu_info["name"] = std::string(prop.name);
    gpu_info["major"] = prop.major;
    gpu_info["minor"] = prop.minor;
    gpu_info["SM"] = prop.multiProcessorCount;
    gpu_info["smem_per_SM"] = prop.sharedMemPerMultiprocessor;
    gpu_info["smem_per_block"] = prop.sharedMemPerBlock;
    
    return gpu_info;
}

PYBIND11_MODULE(gpu_info, m) {
    m.def("get_gpu_info", &get_gpu_info, "");
}

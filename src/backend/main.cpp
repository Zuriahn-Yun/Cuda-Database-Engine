// This file contains a Hardware check

#include <string> 
#include <vector>  
// the nvcc compiler is much slower than c++ compiler
// to recognize this file you need to run this with nvcc compiler
// nvcc -x cu main.cpp -o main -lcudart
// ./main 
#include <iostream> 

#include "Schema.hpp"
#include "DeviceManager.cu"

// Check for Cuda GPU and Cuda Support
#ifdef __CUDACC__
    #include <cuda_runtime.h>
    // If we have nvcc then the Variable Has_Cuda_Support is True
    #define HAS_CUDA_SUPPORT true
#else
    #define HAS_CUDA_SUPPORT false
#endif

// Check for CPU Detection
#ifdef _WIN32
    #include <windows.h>
#else
    #include <unistd.h>
#endif

void checkHardware(){
    std::cout << "-- System Hardware Report -- " << std::endl;
    // 1. CPU Info User will Always have this
    std::cout <<"[CPU] Detection: " << std::endl;
    #ifdef _WIN32
        SYSTEM_INFO sysInfo;
        GetSystemInfo(&sysInfo);
        std::cout << "  - Logical Cores: " << sysInfo.dwNumberOfProcessors << std::endl;
    #else
        long cores = sysconf(_SC_NPROCESSORS_ONLN);
        std::cout << "  - Logical Cores: " << cores << std::endl;
    #endif
    // 2. GPU INFO (CUDA)
    std::cout << "\n[GPU] Detection (CUDA):" << std::endl;
    
    if (HAS_CUDA_SUPPORT) {
        #ifdef __CUDACC__
            // Count Number of GPU's
            // Currently only Support 1 GPU
            int deviceCount = 0;

            cudaError_t err = cudaGetDeviceCount(&deviceCount);
            // 
            if (err != cudaSuccess || deviceCount == 0) {
                std::cout << "  - Status: CUDA compiled, but no physical GPU found." << std::endl;
            } else {
                std::cout << "  - Status: " << deviceCount << " GPU(s) found." << std::endl;
                for (int i = 0; i < deviceCount; ++i) {
                    cudaDeviceProp prop;
                    cudaGetDeviceProperties(&prop, i);
                    std::cout << "  - Device " << i << ": " << prop.name << std::endl;
                    std::cout << "    - Compute: " << prop.major << "." << prop.minor << std::endl;
                    
                    size_t free, total;
                    cudaMemGetInfo(&free, &total);
                    std::cout << "    - VRAM: " << total / (1024 * 1024) << " MB" << std::endl;
                    // Frequency at which the VRAM Operates
                    std::cout << "    - Memory Clock Rate: " << prop.memoryClockRate / 1000 << " MHz" << std::endl;
                    // Larger the Width allows the GPU to move more bits of data simultaneously in a single clock cycle
                    // A clock cycle is how long it takes for a gpu to go from state 0 to 1, every tick we can perform a tiny operation
                    std::cout << "    - Memory Bus Width: " << prop.memoryBusWidth << " bits" << std::endl;
                    // Measured in GB/s actual speed at which data is moved from VRAM into the GPU registers.
                    // This calculates the theoretical peak
                    std::cout << "    - Peak Memory Bandwidth: " << 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1e6 << " GB/s" << std::endl;
                }
            }
        #endif
    } else {
        std::cout << "  - Status: Binary compiled without CUDA support (CPU-only)." << std::endl;
    }
}
int main() {

    checkHardware();
    DeviceManager manager;
    Table users;
    
    // 1. Define structure
    users.addColumn("age", DataType::INT, 1000000);
    
    // 2. Allocate via backend
    manager.allocateColumn(users.columns[0]);
    
    // 3. Decision Engine check
    if(manager.requestGPUExecution(users.columns[0].rowCount)) {
        std::cout << "Optimized for GPU execution." << std::endl;
    }
    
    return 0;
}


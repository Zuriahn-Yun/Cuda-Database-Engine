#include <string> // To add String anmes
#include <vector>  // To hold lists of data
// the nvcc compiler is much slower than c++ compiler
// to recognize this file you need to run this with nvcc compiler
// nvcc -x cu main.cpp -o main -lcudart
// ./main 
#include <iostream> 

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
            std::cout << "Hello World" << std::endl;

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
                }
            }
        #endif
    } else {
        std::cout << "  - Status: Binary compiled without CUDA support (CPU-only)." << std::endl;
    }
}
int main() {
    checkHardware();

    std::cout << "\n[ENGINE] Ready for Columnar Allocation." << std::endl;
    
    // 
    
    return 0;
}


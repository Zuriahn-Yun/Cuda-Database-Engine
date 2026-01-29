#include "Schema.hpp"
#include <cuda_runtime.h>

class DeviceManager {
public:
    // Memory Infrastructure
    void allocateColumn(Column& col) {
        size_t size = col.rowCount * (col.type == DataType::INT ? sizeof(int) : sizeof(float));
        
        // Use Unified Memory: Accessible by both CPU and GPU
        cudaMallocManaged(&col.data, size);
        
        // Optional: Advise the driver that this data will be accessed by the GPU
        int device;
        cudaGetDevice(&device);
        cudaMemAdvise(col.data, size, cudaMemAdviseSetPreferredLocation, device);
    }

    void freeColumn(Column& col) {
        if (col.data) cudaFree(col.data);
    }
    
    // The Decision Engine logic
    bool requestGPUExecution(size_t rows) {
        return rows > 10000; // Your Phase 1 heuristic
    }
};
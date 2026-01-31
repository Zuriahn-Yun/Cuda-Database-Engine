#pragma once

#include <cuda_runtime.h>
#include <iostream>
#include "Schema.hpp"

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

    // Annihalte the Database
    void clearTable(Table& table) {
    std::cout << "[CLEANUP] Freeing VRAM for table: " << table.tableName << std::endl;
    
    for (auto& col : table.columns) {
        if (col.data != nullptr) {
            cudaFree(col.data); // Wipe the GPU/Unified memory
            col.data = nullptr;
        }
    }
    
    table.columns.clear(); // Clear the column metadata
    table.row_count = 0;   // Reset row count
    std::cout << "[CLEANUP] Table wiped successfully." << std::endl;
}
};
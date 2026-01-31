// Currently Only simple CSV Files (text with no commas other than seperators)
// Using Dictionary encoding to map each word to binary number so it will fit on the GPU and the GPU knows its size
// 
#pragma once

#include "Schema.hpp"
#include "DeviceManager.cu"
#include "Common.hpp"

class DataLoader {
public:
    // Dictionary: <ColumnIndex, <StringValue, IntegerID>>
    std::unordered_map<int, std::unordered_map<std::string, int>> dictionaries;
    // Reverse Dictionary: <ColumnIndex, <IntegerID, StringValue>>
    std::unordered_map<int, std::vector<std::string>> reverseDicts;

    void load(const std::string& filepath, Table& table, DeviceManager& manager) {
        std::ifstream file(filepath);
        std::string line;

        // 1. Get Header
        std::getline(file, line);
        std::stringstream ss(line);
        std::string colName;
        while (std::getline(ss, colName, ',')) {
            // We'll initialize columns as UNKNOWN and sniff later
            table.addColumn(colName, DataType::INT, 0); 
        }

        // 2. Temp CPU Storage (Rows)
        std::vector<std::vector<std::string>> rawRows;
        while (std::getline(file, line)) {
            if (line.empty()) continue;
            std::vector<std::string> row;
            std::stringstream lineSs(line);
            std::string cell;
            while (std::getline(lineSs, cell, ',')) row.push_back(cell);
            rawRows.push_back(row);
        }

        table.row_count = rawRows.size();

        // 3. Encode and Allocate
        for (size_t col = 0; col < table.columns.size(); ++col) {
            table.columns[col].rowCount = table.row_count;
            manager.allocateColumn(table.columns[col]);
            
            int* gpuData = (int*)table.columns[col].data;

            for (size_t row = 0; row < table.row_count; ++row) {
                std::string& val = rawRows[row][col];

                if (isNumber(val)) {
                    gpuData[row] = std::stoi(val);
                } else {
                    // It's a string! Encode it.
                    if (dictionaries[col].find(val) == dictionaries[col].end()) {
                        int newID = dictionaries[col].size();
                        dictionaries[col][val] = newID;
                        reverseDicts[col].push_back(val);
                    }
                    gpuData[row] = dictionaries[col][val];
                }
            }
        }
        std::cout << "[SUCCESS] Loaded " << table.row_count << " rows with Dictionary Encoding." << std::endl;
    }
// Report memory after creating Dictionarys 
void reportMemoryUsage(const Table& table, const DataLoader& loader) {
        size_t totalGPUMemory = 0;
        size_t totalDictionaryMemory = 0;

        // 1. Calculate GPU Column Usage
        for (const auto& col : table.columns) {
            // Each column is row_count * 4 bytes (for INT/Encoded IDs)
            totalGPUMemory += table.row_count * sizeof(int);
        }

        // 2. Calculate Dictionary Usage (CPU Side)
        for (auto const& [colIdx, dict] : loader.dictionaries) {
            for (auto const& [str, id] : dict) {
                // String overhead + the string itself + the map integer
                totalDictionaryMemory += sizeof(std::string) + str.capacity() + sizeof(int);
            }
        }

        // 3. Get actual Hardware VRAM status from CUDA
        size_t free_byte;
        size_t total_byte;
        cudaMemGetInfo(&free_byte, &total_byte);

        double free_gb = (double)free_byte / (1024 * 1024 * 1024);
        double used_gpu_mb = (double)totalGPUMemory / (1024 * 1024);

        std::cout << "\n--- [MEMORY REPORT] ---" << std::endl;
        std::cout << "Table Data (GPU): " << std::fixed << std::setprecision(2) << used_gpu_mb << " MB" << std::endl;
        std::cout << "Dictionaries (CPU): " << (double)totalDictionaryMemory / (1024 * 1024) << " MB" << std::endl;
        std::cout << "Available VRAM:   " << free_gb << " GB / " << (double)total_byte / (1024 * 1024 * 1024) << " GB" << std::endl;
        std::cout << "-----------------------\n" << std::endl;
    }
private:
    bool isNumber(const std::string& s) {
        return !s.empty() && std::all_of(s.begin(), s.end(), ::isdigit);
    }

};
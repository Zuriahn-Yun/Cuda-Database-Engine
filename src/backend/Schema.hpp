// Schema.hpp
#pragma once
#include <string>
#include <vector>

enum class DataType { INT, FLOAT };

struct Column {
    std::string name;
    DataType type;
    size_t rowCount;
    void* data; // Points to Unified Memory (Managed by DeviceManager, Unified memory allows the CPU and GPU to use the same pointer)

    Column(std::string n, DataType t, size_t rows) 
        : name(n), type(t), rowCount(rows), data(nullptr) {}
};

struct Table {
    std::string tableName;
    std::vector<Column> columns;
    size_t row_count;
    void addColumn(std::string name, DataType type, size_t rows) {
        columns.emplace_back(name, type, rows);
    }
};
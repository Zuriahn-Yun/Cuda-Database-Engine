// Schema.hpp, Support INTS, FLOATS, STRINGS and BOOLEANS
#pragma once
#include "Common.hpp"

enum class DataType { INT, FLOAT, STRING, BOOL };


struct Column {
    std::string name;
    DataType type;
    size_t rowCount;
    void* data; 

    // This is the constructor the compiler is looking for:
    Column(std::string n, DataType t, size_t rows) 
        : name(n), type(t), rowCount(rows), data(nullptr) {}
};

struct Table {
    std::string tableName;
    std::vector<Column> columns;
    size_t row_count;

    void addColumn(std::string name, DataType type, size_t rows) {
        // This call triggers the "construct" error if the Column constructor above is missing
        columns.emplace_back(name, type, rows);
    }
};
#ifndef SCHEMA_HPP
#define SCHEMA_HPP

#include <string>
#include <vector>

// This tells the engine what KIND of data we are looking at
enum class ColumnType {
    INT,
    FLOAT
};

// This is the blueprint for a single "Column" of data
struct ColumnSchema {
    std::string name;
    ColumnType type;
    size_t length;     // Number of rows
    void* dataPtr;     // The "Handle" or address in memory
};

// This represents a "Table" (A collection of columns)
struct TableSchema {
    std::string tableName;
    std::vector<ColumnSchema> columns;
};

#endif
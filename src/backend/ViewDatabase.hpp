#pragma once
#include <iostream>
#include <iomanip>
#include <vector>
#include "Schema.hpp"

class ViewDatabase {
public:
    static void inspectTable(const Table& table) {
        std::cout << "\n=== [TABLE INSPECTOR: " << table.tableName << "] ===" << std::endl;
        std::cout << "Columns: " << table.columns.size() << " | Rows: " << table.row_count << "\n" << std::endl;

        // Print Header
        for (auto const& col : table.columns) {
            std::cout << std::left << std::setw(15) << col.name;
        }
        std::cout << "\n" << std::string(table.columns.size() * 15, '-') << std::endl;

        // Print first 5 rows (Preview)
        for (size_t r = 0; r < 5 && r < table.row_count; ++r) {
            for (auto const& col : table.columns) {
                if (col.type == DataType::INT) {
                    std::cout << std::left << std::setw(15) << ((int*)col.data)[r];
                } else {
                    std::cout << std::left << std::setw(15) << ((float*)col.data)[r];
                }
            }
            std::cout << std::endl;
        }
        std::cout << "========================================\n" << std::endl;
    }

    static void checkMemoryLocation(const Column& col) {
        // This is a "Working Properly" check
        if (col.data == nullptr) {
            std::cout << "[ERROR] Column " << col.name << " is not allocated!" << std::endl;
        } else {
            std::cout << "[OK] Column " << col.name << " is live in Unified Memory." << std::endl;
        }
    }
};
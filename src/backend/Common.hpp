#pragma once

// All required libraries 

// Standard C++ Libraries
#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <memory>

// CUDA Runtime (Available to both .cpp and .cu files)
#include <cuda_runtime.h>

// Your Project Headers
#include "Schema.hpp"

// Hardware Detection Macro
#ifdef __CUDACC__
    #define IS_CUDA_COMPILER true
#else
    #define IS_CUDA_COMPILER false
#endif
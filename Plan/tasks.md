# Cuda Database Engine: Development Roadmap

### Phase 1: Foundation & Memory Management
*Goal: Build the Smart Allocator and the C++ structure to handle dual-hardware execution.*
- [] **Architecture Design**
    - [x] Define the Columnar Storage Format (optimized for GPU memory coalescing).
    - [x] Create the `DeviceManager` class to detect CUDA capability and available VRAM.
- [ ] **The Decision Engine (CPU vs. GPU)**
    - [ ] Implement a cost-estimation heuristic: $Time_{Transfer} + Time_{GPU\_Exec} < Time_{CPU\_Exec}$.
    - [ ] Logic to handle small datasets on CPU to avoid PCIe overhead.
- [ ] **Memory Infrastructure**
    - [ ] Wrap `cudaMallocManaged` for Unified Memory access.
    - [ ] Implement asynchronous pre-fetching to move data to GPU before the query starts.


---

### Phase 2: CUDA Kernel Development (The SQL Engine)
*Goal: Port core SQL operations to CUDA C++ kernels.*

- [ ] **Parallel Filtering (WHERE)**
    - [ ] Write a kernel for parallel string and numeric comparison.
    - [ ] Use NVIDIA CUB or Thrust for stream compaction (removing rows that don't match the filter).
- [ ] **Aggregation Kernels (SUM, AVG, COUNT)**
    - [ ] Implement parallel reduction trees for high-speed math.
- [ ] **Join Operations**
    - [ ] Build a GPU Hash Join (storing the hash table in shared memory).
- [ ] **Sorting (ORDER BY)**
    - [ ] Integrate a Radix Sort kernel for parallel sorting.



---

### Phase 3: SQLite Compatibility & Persistence
*Goal: Enable the engine to handle files and standard SQL syntax.*

- [ ] **SQL Parser Integration**
    - [ ] Integrate `libhsql` or a similar C++ parser to convert SQL strings into an Abstract Syntax Tree (AST).
- [ ] **SQLite File Loader**
    - [ ] Write a utility to read `.sqlite` or `.db` files into the GPU-optimized memory format.
- [ ] **Disk I/O**
    - [ ] Implement a write-back system to persist GPU-calculated results to disk.

---

### Phase 4: Local LLM Integration
*Goal: Native AI capabilities within the database engine.*

- [ ] **Vector Extension**
    - [ ] Add support for `VECTOR` data types in the schema.
    - [ ] Write a CUDA kernel for Cosine Similarity to enable semantic search.
- [ ] **Inference Pipeline**
    - [ ] Link `llama.cpp` or a custom GGUF loader to the engine.
    - [ ] Create a custom SQL keyword: `SELECT GENERATE_TEXT(prompt_col) FROM table`.
- [ ] **VRAM Orchestrator**
    - [ ] Implement a memory-sharing protocol so the LLM weights and SQL data do not cause Out-of-Memory (OOM) errors.

---

### Phase 5: Testing & Benchmarking
- [ ] **Unit Tests:** Validate CPU and GPU results match exactly (Bit-for-bit parity).
- [ ] **Benchmarking:** Compare execution time vs. standard SQLite for datasets greater than 1GB.
- [ ] **Stress Test:** Ensure the hardware switcher handles rapid-fire queries without leaking VRAM.
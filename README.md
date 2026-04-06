A high-performance C++ database engine utilizing CUDA to accelerate SQL operations through columnar storage and parallel kernel execution. The system features a hybrid decision engine that intelligently offloads heavy analytical queries to the GPU while handling smaller tasks on the CPU. It is built for massive datasets, leveraging Unified Memory for seamless data sharing and high-speed throughput.

## Future Implementations

| ID | Feature | Description | Group |
|----|---------|-------------|-------|
| A1 | Async Prefetching | Use `cudaMemPrefetchAsync()` to move column data to GPU before kernel launch, hiding PCIe latency | Phase 1 Completion |
| A2 | Improved Decision Engine | Replace the row-count stub with a real cost model estimating transfer time vs kernel execution time | Phase 1 Completion |
| A3 | Column Statistics | Track min, max, null count, and distinct count per column at load time for query optimization | Phase 1 Completion |
| B1 | Parallel Filter (WHERE) | CUDA kernel producing a boolean mask for row predicates, with CUB stream compaction | CUDA Kernels |
| B2 | Aggregation Kernels | Parallel reduction kernels for SUM, AVG, COUNT, MIN, MAX using shared memory | CUDA Kernels |
| B3 | ORDER BY (Sorting) | Thrust `sort_by_key` or custom radix sort kernel for result ordering | CUDA Kernels |
| B4 | GROUP BY | GPU hash map assigning rows to buckets, then aggregating per bucket | CUDA Kernels |
| B5 | Hash Join | Build/probe hash table in shared memory for multi-table queries | CUDA Kernels |
| C1 | C++ Query Builder API | Fluent C++ API (`Query(table).filter(...).aggregate(...)`) for using kernels before SQL parsing | Query & SQL |
| C2 | SQL Parser Integration | Integrate `libhsql` to convert SQL strings into an AST mapped to the Query API | Query & SQL |
| C3 | Interactive REPL | Terminal loop for typing SQL queries and printing results interactively | Query & SQL |
| D1 | CSV Export | Write query results back to CSV with dictionary decoding for string columns | Persistence & I/O |
| D2 | SQLite File Loader | Load existing `.sqlite`/`.db` files directly into the columnar GPU format | Persistence & I/O |
| D3 | Binary Persistence | Serialize the in-memory columnar format to disk for cross-session persistence | Persistence & I/O |
| E1 | Query Profiler | CUDA events timing each stage (transfer, kernel, fetch) with per-query breakdown | Performance |
| E2 | Result Caching | Cache query results keyed by query hash and data version to avoid redundant execution | Performance |
| E3 | Run-Length Encoding | Compress sorted/repeated integer columns, especially effective after dictionary encoding | Performance |
| E4 | Multi-GPU Support | Partition tables across multiple GPUs and aggregate partial results | Performance |
| F1 | Vector Type + Cosine Similarity | Add `VECTOR(n)` column type with a batch cosine similarity CUDA kernel for embedding search | LLM / Vector |
| F2 | VRAM Orchestrator | Budget VRAM between LLM weights and SQL column data to prevent OOM | LLM / Vector |
| F3 | GENERATE_TEXT() Extension | SQL keyword that pipes column data into a loaded GGUF model for inline text generation | LLM / Vector |

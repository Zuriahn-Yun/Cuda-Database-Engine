# Questions
What will the GPU do better than the CPU since it has thousands of threads. \
How can we optimize the GPU to use all of its threads to move and store data as quickly as possible? \
Do we still want to employ the CPU in order to do some tasks that require threads that move much quicker? Or do we only want to use the GPU? \

# Vocab
Heterogeneous systems - Combine GPUS, CPU and other processors into one system to optimize performance
SM - Streaming Multiprocessor
PTX - Parallel Thread Execution
ISA - Instruction Set Architecture
Cubins - 
Fatbins - 
Library Binarys - Pre Compiled non readable code that is used to perform specific tasks.
Memory coalescing - When a warp requests memory the gpu is faster if those threads are acccesing neighboring memory addresses.
Warp - A group of 32 Threads


# Compute Compatibility 
12.0 
Major version: 12
Minor Version: 0
Shows us all the possbile features availible to us with this compute Compatibility.
https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html#compute-capabilities-table-features-and-technical-specifications-feature-support-per-compute-capability

Which features might be useful for this project? Why? 

Cuda Pipelines and accelerated Pipelines? 
https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/pipelines.html#pipelines


# Cuda Runtime API and Cuda Driver API
---

# Cuda PTX (Parallel Thread Execution)
What is this? 
High level assembly language for GPU's
Where is it? Why is it required? What does it do? 
https://docs.nvidia.com/cuda/parallel-thread-execution/index.html
What is ISA and why is it important? (Instruction Set Architecture)
What do these do? 
They essentially allows Cuda/C++ code to be usable for future generations of GPU's.
Do we need to know more about this? Is this important? Do we want our projects to be usable in the future? 

# Cubins and Fatbins
C++ is compiled into PTX and then PTX is compiled into binary for the GPU. GPU binary is collaed Cubin.
How is Cubin different from Tradional binary? 



# Other Referenced Pages
https://github.com/NVIDIA/accelerated-computing-hub

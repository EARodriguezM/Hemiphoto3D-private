#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

// Fatal error check — use after every CUDA call
#define CUDA_CHECK(call)                                                      \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                       \
                    __FILE__, __LINE__, cudaGetErrorString(err));               \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// Check last kernel launch
#define CUDA_CHECK_KERNEL()                                                    \
    do {                                                                       \
        CUDA_CHECK(cudaGetLastError());                                        \
        CUDA_CHECK(cudaDeviceSynchronize());                                   \
    } while (0)

// RAII wrapper for device memory
template <typename T>
struct DeviceBuffer {
    T* ptr = nullptr;
    size_t count = 0;

    void alloc(size_t n) {
        free();
        count = n;
        CUDA_CHECK(cudaMalloc(&ptr, n * sizeof(T)));
    }

    void free() {
        if (ptr) { cudaFree(ptr); ptr = nullptr; count = 0; }
    }

    void upload(const T* host_data, size_t n) {
        if (n > count) alloc(n);
        CUDA_CHECK(cudaMemcpy(ptr, host_data, n * sizeof(T), cudaMemcpyHostToDevice));
    }

    void download(T* host_data, size_t n) const {
        CUDA_CHECK(cudaMemcpy(host_data, ptr, n * sizeof(T), cudaMemcpyDeviceToHost));
    }

    ~DeviceBuffer() { free(); }

    // No copy, allow move
    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& o) noexcept : ptr(o.ptr), count(o.count) { o.ptr = nullptr; o.count = 0; }
    DeviceBuffer& operator=(DeviceBuffer&& o) noexcept { free(); ptr = o.ptr; count = o.count; o.ptr = nullptr; o.count = 0; return *this; }
};

// Query and print GPU info
void printGPUInfo(int device_id = 0);

// Get available GPU memory in bytes
size_t getAvailableGPUMemory(int device_id = 0);

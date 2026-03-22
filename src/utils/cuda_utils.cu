#include "cuda_utils.cuh"
#include <cstdio>

void printGPUInfo(int device_id) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));

    printf("GPU %d: %s\n", device_id, prop.name);
    printf("  Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("  Total memory: %.0f MB\n", prop.totalGlobalMem / (1024.0 * 1024.0));
    printf("  SMs: %d\n", prop.multiProcessorCount);
    printf("  Max threads/block: %d\n", prop.maxThreadsPerBlock);
    printf("  Warp size: %d\n", prop.warpSize);
    printf("  Max shared memory/block: %.0f KB\n", prop.sharedMemPerBlock / 1024.0);
}

size_t getAvailableGPUMemory(int device_id) {
    CUDA_CHECK(cudaSetDevice(device_id));
    size_t free_bytes, total_bytes;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    return free_bytes;
}

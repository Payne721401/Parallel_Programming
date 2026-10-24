// Test-only executable: compile the fixed main unchanged, wrapping only run().
// Profiler collection excludes input generation/hash; trace timings are diagnostic.
#include <cuda_profiler_api.h>
#define run run_traced
#include "../main.cu"
#undef run

void run(uint8_t *grid, const Problem &p);

void run_traced(uint8_t *grid, const Problem &p) {
    cuda_check(cudaProfilerStart(), "start trace");
    run(grid, p);
    cuda_check(cudaDeviceSynchronize(), "finish trace work");
    cuda_check(cudaProfilerStop(), "stop trace");
}

/**
 * 
 * The student is required to add content to this file.  This file is
 * your implementation of the project and will be submitted for grading.
 * 
 */

#include "main.h"
#include "student.h"

/**********************************************************************************
 * 
 * Implement your GPU device kernel(s) here (e.g., the bitonic sort kernel).
 * 
 **********************************************************************************/

DTYPE* d_arr;
int d_size;
// Tracks whether the host input array was successfully page-locked so that
// cleanup only unregisters it on success.
static bool arrCpuPinned = false;

__global__ void compare_exchange_cuda(DTYPE* arr, int i, int j, int d_size) {
    // Pair-index threading: one thread per comparator (d_size/2 threads), not
    // one per element. This launches half as many blocks and removes the old
    // `if (k > p) return` half-block waste (for j >= log2(TILE) the global path
    // strides span whole blocks, so previously half of all blocks launched only
    // to early-return). Map pair index t -> the "low" element by inserting a 0
    // bit at position j, exactly like the shared kernel's low/partner mapping.
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (d_size >> 1)) return; // guard partial final block (small arrays)
    int stride = 1 << j;
    int low = (t / stride) * (2 * stride) + (t % stride);
    int partner = low + stride;
    bool asc = (low & (1 << i)) == 0;
    DTYPE a = arr[low];
    DTYPE b = arr[partner];
    if ((asc && a > b) || (!asc && a < b)) {
        arr[low] = b;
        arr[partner] = a;
    }
}

/**
 * Shared-memory kernel: each block owns a contiguous TILE-element chunk.
 * It loads the chunk into shared memory once, runs every bitonic step from
 * j_start down to 0 locally (all strides < TILE stay inside the chunk), then
 * writes the chunk back once. This collapses j_start+1 global kernel launches
 * into a single launch and removes the per-step global round trips.
 *
 * Grid-stride loops decouple TILE from BLOCK_DIM: each thread processes
 * TILE/BLOCK_DIM elements on load/store and TILE/2/BLOCK_DIM pairs per step.
 */
__global__ void bitonic_shared(DTYPE* arr, int i, int j_start) {
    __shared__ DTYPE tile[TILE];
    int base = blockIdx.x * TILE;

    // grid-stride load: bring the block's chunk into shared memory.
    // NOTE: int4-vectorizing this loop was tried (Trial 8) and REGRESSED the
    // kernel — int4 to shared introduces 4-way bank conflicts (thread t writing
    // tile[4t..4t+3] makes 8 threads span all 32 banks), which outweighs the
    // global-side win. Scalar (thread t -> tile[t]) is conflict-free; kept.
    for (int e = threadIdx.x; e < TILE; e += BLOCK_DIM) {
        tile[e] = arr[base + e];
    }
    __syncthreads();

    int pairs = TILE / 2;
    for (int j = j_start; j >= 0; j--) {
        int stride = 1 << j;
        // grid-stride over the TILE/2 compare-exchange pairs.
        // NOTE: register-blocking this (gather all UNROLL pairs into registers,
        // then compare/store — Trial 11) REGRESSED kernel time (65→74 ms) and
        // shared mem-throughput (47→38%). This kernel is already ~91% occupancy,
        // so warp-level parallelism already hides shared latency; per-thread ILP
        // just raised register pressure, lowered occupancy, and slowed it. The
        // simple per-pair grid-stride loop is best; kept.
        for (int t = threadIdx.x; t < pairs; t += BLOCK_DIM) {
            // map pair index t to the "low" element of its pair within the tile
            int low = (t / stride) * (2 * stride) + (t % stride);
            int partner = low + stride;
            // direction is stage-based, so use the global index
            bool asc = ((base + low) & (1 << i)) == 0;
            DTYPE a = tile[low];
            DTYPE b = tile[partner];
            if ((asc && a > b) || (!asc && a < b)) {
                tile[low] = b;
                tile[partner] = a;
            }
        }
        __syncthreads(); // every step must complete before the next reads
    }

    // grid-stride store: write the sorted chunk back to global memory
    for (int e = threadIdx.x; e < TILE; e += BLOCK_DIM) {
        arr[base + e] = tile[e];
    }
}

__global__ void fill_tail(DTYPE* arr, int start, int end, DTYPE value) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x + start;
    if (idx < end) arr[idx] = value;
}


/**********************************************************************************
 * 
 * Implement your utility functions here
 * 
 **********************************************************************************/

unsigned int next_pow2(unsigned int x) {
    // Effectively calculate this expression but faster: 
    // 1 << (int)ceil(log2((double)size));
    // 1 << (log2(size-1)+1)
    if (x <= 1) return 1; // cant log2(1-1)
    x--; // in case already full 1bit (1111)
    // Fill to full 1bit (1111)
    // power of 2 bit shifts to fill 1 from the highest set bit
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;

    return x+1; // 10000
}


/**********************************************************************************
 * 
 * Implement the three main program functions
 * 
 **********************************************************************************/


/**
 * This function transfers data from Host to Device
 */
void host_to_dev()
{
    d_size = next_pow2(size);
    cudaMalloc((void**) &d_arr, d_size*sizeof(DTYPE));
    // Page-lock the host source (best-effort) so H2D uses the fast DMA path.
    arrCpuPinned = (cudaHostRegister(arrCpu, size*sizeof(DTYPE), cudaHostRegisterDefault) == cudaSuccess);
    if (!arrCpuPinned) cudaGetLastError(); // clear error state if registration failed
    cudaMemcpy(d_arr, arrCpu, size*sizeof(DTYPE), cudaMemcpyHostToDevice);
    // NOTE: tail padding is launched at the start of bitonic_sort() (kernel-timed
    // phase), not here, so the H2D timer measures only the host->device copy.
}

/**
 * This function performs the bitonic sort and merge by calling the
 * kernels you have defined in the section above
 */
void bitonic_sort()
{
    dim3 block_size(BLOCK_DIM);
    // pair-index threading: d_size/2 comparators per phase, so half the threads
    dim3 grid_size(((d_size >> 1) + BLOCK_DIM - 1) / BLOCK_DIM);
    // shared kernel: one block per TILE-element chunk
    dim3 grid_shared(d_size / TILE);

    // Pad the power-of-two tail with INT_MAX before sorting. Done here (not in
    // host_to_dev) so its cost falls in the kernel timer rather than the H2D
    // transfer timer; the sort cannot begin until the sentinels are in place.
    int tail = d_size - size;
    if (tail > 0) {
        dim3 fill_blocks = (tail + BLOCK_DIM - 1) / BLOCK_DIM;
        fill_tail<<<fill_blocks, BLOCK_DIM>>>(d_arr, size, d_size, INT_MAX);
    }

    // A step j keeps its partner (stride 2^j) inside a TILE chunk when
    // 2^(j+1) <= TILE, i.e. j < log2(TILE).
    int log_tile = (int)log2((float)TILE);
    // Only use the shared kernel when the array is at least one full tile.
    bool use_shared = (d_size >= TILE);

    // stage i: construct sorted subsequence of size 2^i from bitonic sequence of size 2^i
    int num_stages = (int)log2((float)d_size);
    for (int i=1; i<=num_stages; i++) {
        for (int j=i-1; j >= 0; j--) {
            if (use_shared && j < log_tile) {
                // remaining steps (j..0) all fit in a tile: finish them in
                // a single shared-memory launch, then move to next stage
                bitonic_shared<<<grid_shared, block_size>>>(d_arr, i, j);
                break;
            }
            compare_exchange_cuda<<<grid_size, block_size>>>(d_arr, i, j, d_size);
        }
    }
}

/**
 * This functiuon transfers the sorted data from Device to Host
 */
DTYPE *dev_to_host()
{
    // malloc + cudaHostRegister lets the copy use the fast pinned DMA path.
    // Measured in-harness: D2H ~64 ms vs ~91 ms for a plain pageable copy. The
    // copy itself is ~8 ms; the rest is the one-time page-fault+lock cost of
    // registering the fresh buffer (cudaMallocHost is worse — ~150 ms to alloc).
    arrSortedGpu = (DTYPE*)malloc(size*sizeof(DTYPE));
    bool reg = (cudaHostRegister(arrSortedGpu, size*sizeof(DTYPE), cudaHostRegisterDefault) == cudaSuccess);
    if (!reg) cudaGetLastError();
    cudaMemcpy(arrSortedGpu, d_arr, size*sizeof(DTYPE), cudaMemcpyDeviceToHost);
    if (reg) cudaHostUnregister(arrSortedGpu);
    return arrSortedGpu;
}

/**
 * This function frees memory and anything else the student requires 
 * before exiting the program
 */
void cleanup(){
    
    // You may modify/remove these as needed to make your implementation work
    // properly. The defaults provided here allow the skeleton code to compile.    
    cudaFree(d_arr);
    if (arrCpuPinned) cudaHostUnregister(arrCpu);
    free(arrCpu);
    free(arrSortedGpu);
}

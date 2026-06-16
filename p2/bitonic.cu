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

__global__ void compare_exchange_scalar_cuda(DTYPE* arr, int i, int j, int d_size) {
    // Pair-index threading: one thread per comparator (d_size/2 threads), not
    // one per element. This launches half as many blocks and removes the old
    // `if (k > p) return` half-block waste (for j >= log2(TILE) the global path
    // strides span whole blocks, so previously half of all blocks launched only
    // to early-return). Map pair index t -> the "low" element by inserting a 0
    // bit at position j, exactly like the shared kernel's low/partner mapping.
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (d_size >> 1)) return; // guard partial final block (small arrays)
    int stride = 1 << j;
    // map pair index t -> low element by inserting a 0 bit at position j.
    // Bit-shift form of (t/stride)*(2*stride) + (t%stride); avoids runtime
    // integer div/mod (stride is not a compile-time constant).
    int low = ((t >> j) << (j + 1)) | (t & (stride - 1));
    int partner = low + stride;
    bool asc = (low & (1 << i)) == 0;
    DTYPE a = arr[low];
    DTYPE b = arr[partner];
    // branchless compare-exchange (min/max + select on direction)
    DTYPE lo = min(a, b);
    DTYPE hi = max(a, b);
    arr[low]     = asc ? lo : hi;
    arr[partner] = asc ? hi : lo;
}

__device__ __forceinline__ uint32_t pack_u16_pair(uint32_t low, uint32_t high) {
    return (high << 16) | low;
}

__global__ void compare_exchange_cuda(DTYPE* arr, int i, int j, int d_size) {
    // Packed uint16 path: for j >= 1, two adjacent low elements and their two
    // adjacent partners are both 32-bit aligned and have the same direction.
    int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= (d_size >> 2)) return;

    int t = q << 1;
    int stride = 1 << j;
    int low = ((t >> j) << (j + 1)) | (t & (stride - 1));
    int partner = low + stride;

    uint32_t a_pack = *reinterpret_cast<const uint32_t*>(arr + low);
    uint32_t b_pack = *reinterpret_cast<const uint32_t*>(arr + partner);

    uint32_t a0 = a_pack & 0xFFFFu;
    uint32_t a1 = a_pack >> 16;
    uint32_t b0 = b_pack & 0xFFFFu;
    uint32_t b1 = b_pack >> 16;

    uint32_t lo0 = (a0 < b0) ? a0 : b0;
    uint32_t hi0 = (a0 < b0) ? b0 : a0;
    uint32_t lo1 = (a1 < b1) ? a1 : b1;
    uint32_t hi1 = (a1 < b1) ? b1 : a1;

    bool asc = (low & (1 << i)) == 0;
    uint32_t low_pack = asc ? pack_u16_pair(lo0, lo1) : pack_u16_pair(hi0, hi1);
    uint32_t partner_pack = asc ? pack_u16_pair(hi0, hi1) : pack_u16_pair(lo0, lo1);

    *reinterpret_cast<uint32_t*>(arr + low) = low_pack;
    *reinterpret_cast<uint32_t*>(arr + partner) = partner_pack;
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
    __shared__ int tile[TILE];
    int base = blockIdx.x * TILE;

    // grid-stride load: bring the block's chunk into shared memory.
    // NOTE: int4-vectorizing this loop was tried (Trial 8) and REGRESSED the
    // kernel — int4 to shared introduces 4-way bank conflicts (thread t writing
    // tile[4t..4t+3] makes 8 threads span all 32 banks), which outweighs the
    // global-side win. Scalar (thread t -> tile[t]) is conflict-free; kept.
    for (int e = threadIdx.x; e < TILE; e += BLOCK_DIM) {
        tile[e] = (int)arr[base + e];
    }
    __syncthreads();

    int pairs = TILE / 2;
    for (int j = j_start; j >= 0; j--) {
        int stride = 1 << j;
        // grid-stride over the TILE/2 compare-exchange pairs.
        for (int t = threadIdx.x; t < pairs; t += BLOCK_DIM) {
            // map pair index t to the "low" element of its pair within the tile.
            // Bit-shift form of (t/stride)*(2*stride) + (t%stride): insert a 0 bit at position j.
            int low = ((t >> j) << (j + 1)) | (t & (stride - 1));
            int partner = low + stride;
            // direction is stage-based, so use the global index
            bool asc = ((base + low) & (1 << i)) == 0;
            DTYPE a = tile[low];
            DTYPE b = tile[partner];
            // branchless compare-exchange (min/max + select on direction)
            DTYPE lo = min(a, b);
            DTYPE hi = max(a, b);
            tile[low]     = asc ? lo : hi;
            tile[partner] = asc ? hi : lo;
        }
        __syncthreads(); // every step must complete before the next reads
    }

    // grid-stride store: write the sorted chunk back to global memory
    for (int e = threadIdx.x; e < TILE; e += BLOCK_DIM) {
        arr[base + e] = (DTYPE)tile[e];
    }
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
    // packed global kernel: two adjacent comparators per thread
    dim3 grid_packed(((d_size >> 2) + BLOCK_DIM - 1) / BLOCK_DIM);
    // shared kernel: one block per TILE-element chunk
    dim3 grid_shared(d_size / TILE);

    // Pad the power-of-two tail
    int tail = d_size - size;
    if (tail > 0) {
        cudaMemset(d_arr + size, 0xFF, (size_t)tail * sizeof(DTYPE));
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
            if (j > 0) {
                compare_exchange_cuda<<<grid_packed, block_size>>>(d_arr, i, j, d_size);
            } else {
                compare_exchange_scalar_cuda<<<grid_size, block_size>>>(d_arr, i, j, d_size);
            }
        }
    }
}

/**
 * This functiuon transfers the sorted data from Device to Host
 */
DTYPE *dev_to_host()
{
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

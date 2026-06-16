# CS 7295 GPU HW & SW — Project 2: Parallel Bitonic Sort

**Author:** *Kyle Nguyen* | **GTID:** *903953383* | **Date:** *2026-06-15*

---

## 1. Introduction and Problem Overview

This project implements a parallel integer sorting routine in CUDA based on
**Bitonic Sort**, a comparison-network algorithm that is particularly well
suited to GPU execution. Unlike mergesort, whose later merge stages collapse
onto a shrinking number of active threads, bitonic sort performs a fixed,
data-independent sequence of compare-exchange operations in which **every
element is active at every step**. This regularity eliminates load imbalance
and divergent control flow, making the algorithm map cleanly onto the SIMT
execution model of the NVIDIA H100.

The work is organized in two parts. **Task 1** is a straightforward global-memory
implementation that issues one kernel launch per bitonic sub-stage. **Task 2**
optimizes this baseline by introducing a **shared-memory kernel** that fuses the
small-stride steps of each merge into a single launch, drastically reducing both
the number of kernel launches and the volume of global-memory traffic.

## 2. Algorithm Background

Bitonic sort builds a sorted array bottom-up. For an array of length
$n = 2^m$, the outer loop runs stages $i = 1 \dots \log_2 n$. Stage $i$
turns adjacent sorted runs of length $2^{i-1}$ into bitonic runs of length
$2^i$ and then merges each into a sorted run of length $2^i$. The merge for
stage $i$ consists of $i$ sub-steps $j = i-1 \dots 0$, each performing a
compare-exchange between every element $k$ and its partner $k \oplus 2^j$.

The sort direction for a comparator is determined purely by position:

```
asc = ((k & (1 << i)) == 0)
```

so even-indexed chunks of length $2^i$ sort ascending and odd chunks descending,
producing the bitonic input required by the following stage. Because each of the
$\frac{1}{2}\log_2 n (\log_2 n + 1)$ sub-steps touches all $n/2$ comparator
pairs, total work is $O(n \log^2 n)$ with parallel depth $O(\log^2 n)$. The
extra $\log n$ factor over a work-optimal sort is the price paid for perfect
regularity and parallelism.

## 3. Implementation

### 3.1 Data Preparation and Padding

Bitonic sort requires a power-of-two length. The host routine `host_to_dev()`
computes the next power of two with a branchless bit-fill `next_pow2()`,
allocates the device buffer, and copies the input. The padded tail is filled
**on the device** with `INT_MAX` by the `fill_tail` kernel, so the sentinel
values sort to the high end and never disturb the valid data. All allocation
and initialization happens inside the timed region, in compliance with the
project rules (no static/global pre-initialization).

### 3.2 Task 1 — Global-Memory Kernel

The baseline kernel `compare_exchange_cuda` assigns one thread per array index
$k$. Each thread computes its partner $p = k \oplus 2^j$ and performs the
compare-exchange only when $k < p$, which guarantees each pair is processed
exactly once and avoids write races. The host issues one launch per
$(i, j)$ sub-step:

```
for i = 1 .. log n
    for j = i-1 .. 0
        compare_exchange_cuda<<<grid, block>>>(arr, i, j, n)
```

This is correct but launch-bound: a 100M-element array needs
$\frac{1}{2}\log^2 n \approx 350$ kernel launches, and **every** comparator
reads two values from global memory and writes two back.

### 3.3 Task 2 — Shared-Memory Kernel

The key observation is that when the comparator stride $2^j$ is smaller than the
chunk of elements a block owns, the partner $k \oplus 2^j$ always lands inside
the same chunk. Those steps therefore need no cross-block communication and can
be executed entirely in **shared memory**.

`bitonic_shared` gives each block a contiguous `TILE`-element chunk. It:

1. **Loads** the chunk into a `__shared__` buffer with a grid-stride loop.
2. **Runs every step** from `j_start` down to `0` locally, with a
   `__syncthreads()` between steps so all comparators of one step complete
   before the next reads the data.
3. **Stores** the chunk back to global memory once.

Each thread maps its pair index `t` to the low element of its comparator with
`low = (t / stride) * (2*stride) + (t % stride)`, which assigns exactly one
comparator per thread and removes the wasted half-warp and the `if (k > p)`
divergence present in the global kernel. The sort direction still uses the
**global** index `base + low`, preserving correctness across the block boundary.

### 3.4 Dispatch Logic

`bitonic_sort()` keeps the global kernel for the large-stride steps
($2^j \ge \text{TILE}$, where partners cross block boundaries) and switches to a
single `bitonic_shared` launch the moment `j` drops below `log2(TILE)`, which
finishes all remaining steps `j .. 0` of that stage in one launch. For small
inputs (below one tile) the code falls back to the all-global path, keeping the
2K and 10K correctness cases robust.

The effect on launch count is substantial: each stage's inner loop of up to
`log2(TILE)` global launches is replaced by **one** shared-memory launch,
removing the majority of launches and global round trips at scale.

### 3.5 Tunable Parameters

`student.h` exposes two compile-time knobs so the configuration can be swept
with a recompile:

| Macro | Meaning | Effect |
|-------|---------|--------|
| `BLOCK_DIM` | threads per block | occupancy / scheduling granularity |
| `TILE` | elements per block in shared kernel | global-traffic reduction vs. shared-mem pressure |

Grid-stride load/store loops decouple `TILE` from `BLOCK_DIM`, so each can be
tuned independently. Larger `TILE` absorbs more steps per launch (better memory
throughput and meps) but raises shared-memory pressure, which lowers occupancy.

## 4. Performance Evaluation

### 4.1 Methodology

**Test hardware.** All measurements in this report were taken on a **PACE-ICE
H100** node (NVIDIA H100 80 GB HBM3, compute capability 9.0) — the same hardware
used for grading. The build flags match the grading script
(`nvcc -Xcompiler -rdynamic -lineinfo`).

Correctness is checked across the graded sizes (2K, 10K, 100K, 1M, 10M) via the
program's `FUNCTIONAL SUCCESS` output. Throughput is measured at 100M elements;
**Achieved Occupancy** and **Memory Throughput** are measured at 10M with NSight
Compute:

```
ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active \
    --print-summary per-gpu ./a.out 10000000
ncu --metrics gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed \
    --print-summary per-gpu ./a.out 10000000
```

### 4.2 Parameter Sweep

The shared-memory tile size `TILE` was swept across {1024, 2048, 4096, 8192}
with `BLOCK_DIM = 256` on the PACE-ICE H100. Throughput was measured at 100M
elements (best of three runs, per the grading protocol) and the NSight metrics
at 10M. Results:

| TILE | Shared mem/block | Occupancy @10M | Mem Thpt @10M | Best meps @100M | Kernel time @100M | Grade score |
|------|------------------|----------------|---------------|-----------------|-------------------|-------------|
| 1024 | 4 KB  | 56.22% | 45.84% | 465.4 | 80.87 ms | 16.79 |
| 2048 | 8 KB  | 55.55% | 47.28% | 482.4 | 73.09 ms | 16.89 |
| 4096 | 16 KB | 54.58% | 47.55% | 494.8 | 67.57 ms | 16.89 |
| 8192 | 32 KB | 46.40% | 46.86% | 506.2 | 63.95 ms | 16.90 |

### 4.3 Analysis of the Sweep

**Kernel time** is the cleanest signal and improves monotonically as `TILE`
grows, from 80.9 ms at 1024 down to 64.0 ms at 8192. This is the expected payoff
of step fusion: a larger tile absorbs more of each merge's small-stride steps
into a single shared-memory launch, eliminating global round trips and
kernel-launch overhead. The marginal gain shrinks at each doubling (≈7.8, 5.5,
and 3.6 ms), showing diminishing returns as the remaining work becomes dominated
by the unavoidable large-stride global passes. Every configuration from 2048 up
clears the 80 ms Option-2 threshold for the full 10/10 kernel-time score.

**meps** rises monotonically with `TILE` in lock-step (465 → 482 → 495 → 506),
tracking the kernel-time trend because the H100's D2H transfer is stable here
(~91–93 ms on the best run of each batch), so faster kernels translate directly
into higher end-to-end throughput. All four configs remain below the 900-meps
floor required for grade Option 1, so the Option-2 kernel+transfer path sets the
grade for each.

**Occupancy** holds near 55% through `TILE = 4096` but falls to 46.4% at 8192,
where 32 KB of shared memory per block limits the number of resident blocks per
SM — the per-kernel data in Section 5 shows this hits the shared kernel itself,
whose occupancy drops from ~91% at 4096 to ~66% at 8192. **Memory throughput**
peaks at `TILE = 4096` (47.6%) and dips slightly at 8192, consistent with the
occupancy loss reducing the SM's ability to hide memory latency even though raw
global traffic keeps falling.

**Configuration choice.** The grade is effectively tied from 2048 up
(16.89–16.90), so the tie-breaker is the occupancy/throughput profile.
`TILE = 8192` posts the best kernel time (64.0 ms) and meps (506) but pays an
8-point occupancy cliff for only a 3.6 ms kernel-time gain. **`TILE = 4096` is
selected** as the final configuration: it sits at the knee of the curve with the
highest memory throughput of the sweep (47.6%), retains ~55% aggregate (and ~91%
shared-kernel) occupancy, and concedes only 3.6 ms of kernel time versus 8192.
All four configurations passed functional correctness on every graded size.

### 4.4 Selected Configuration and Observations Relative to Targets

The chosen `TILE = 4096` build, measured at 100M elements (best of three runs)
on the PACE-ICE H100, posts:

| Component | Value | Score |
|-----------|-------|-------|
| Kernel time @100M | 67.6 ms | 10.0 / 10 |
| Memory transfer (H2D + D2H) @100M | 134.5 ms | 0.89 / 4 |
| meps @100M | 494.8 | — (Option 2 used) |
| Achieved Occupancy @10M | 54.58% | 0 / 1 (target 65%) |
| Memory Throughput @10M | 47.55% | 0 / 1 (target 75%) |
| **Total grade** | | **16.89 / 20** |

Kernel time of **67.6 ms** comfortably clears the 80 ms Option-2 ceiling for the
**full 10/10 kernel-time score**; since the 494.8 meps is below the 900 floor for
Option 1, the Option-2 kernel+transfer path sets the grade. All graded sizes
report `FUNCTIONAL SUCCESS`.

Achieved Occupancy (54.6%) and Memory Throughput (47.6%) remain short of the
65% / 75% thresholds that earn the two extra-credit points. As the Section 5
breakdown shows, this is not a shared-kernel deficiency — that kernel reaches
~91% occupancy — but the average of the high-occupancy shared kernel with the
lower-occupancy global passes and the one-shot padding kernel. Raising the
headline numbers would require restructuring the global-pass kernel, not faster
hardware. The combined memory-transfer time of 134.5 ms (≈93 ms of it D2H) yields
only 0.89 / 4, so reducing D2H transfer is the single largest remaining lever on
the overall grade of **16.89 / 20**.

## 5. Profiler Analysis

NSight Compute was run at 10M elements on the **PACE-ICE H100** with per-kernel
summaries for the selected `TILE = 4096` build. The grade-script aggregates
(54.6% occupancy, 47.6% memory throughput) are the arithmetic mean across the
three kernels; the per-kernel breakdown below explains what drives them.

**Achieved Occupancy** (`sm__warps_active.avg.pct_of_peak_sustained_active`):

| Kernel | Occupancy | Invocations |
|--------|-----------|-------------|
| `bitonic_shared` | 90.92% | 24 |
| `compare_exchange_cuda` | 53.88% | 78 |
| `fill_tail` | 17.67% | 1 |

**Memory Throughput** (`gpu__compute_memory_throughput...`):

| Kernel | Mem Throughput | Invocations |
|--------|----------------|-------------|
| `bitonic_shared` | 51.08% | 24 |
| `compare_exchange_cuda` | 56.55% | 78 |
| `fill_tail` | 34.96% | 1 |

The breakdown is the key insight: the **shared-memory kernel itself reaches ~91%
occupancy**, well above the 65% target. The reported aggregate (54.6%) is pulled
down by the global-pass kernel (53.9%) and the one-shot `fill_tail` padding
kernel (17.7%). Since grading averages across kernels, the global passes — not
the shared kernel — are what hold the headline occupancy below target.

**Memory access counts**
(`l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum`,
`l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum`):

- Every launch of either compute kernel issues **2,097,152 global-load sectors**
  — exactly one full pass over the padded 16M-element (64 MB) array
  ($16\text{M} \times 4\,\text{B} / 32\,\text{B per sector}$). Both kernels read
  the array once per launch; the shared kernel's advantage is therefore **fewer
  launches** (24 vs. 78), since each shared launch fuses several merge steps
  internally, not fewer sectors per launch.
- **Local-memory loads are 0** for all kernels, confirming no register spilling.

**Divergence**
(`smsp__thread_inst_executed_per_inst_executed.ratio`): the measured ratio is
**32.0** (full warp) for *both* compute kernels, i.e. no warp divergence in
either. The companion predicated-on ratio
(`smsp__thread_inst_executed_pred_on_per_inst_executed.ratio`) is 27.2 for
`bitonic_shared` and 29.1 for `compare_exchange_cuda` — ~85–91% of the 32 lanes
are active per instruction, confirming the `if (k > p) return` comparator guard
masks only a small minority of lanes through **predication rather than a
divergent branch**. The shared kernel's one-comparator-per-thread mapping
therefore matches — not beats — the baseline on divergence; its gains come from
reduced launches and global traffic, not from removing divergence.

## 6. Discussion of Optimizations and Effectiveness

| Optimization | Rationale | Effectiveness |
|--------------|-----------|---------------|
| Device-side padding (`fill_tail`) | avoids host loop and keeps prep on GPU | enables non-power-of-two inputs at negligible cost |
| One thread per unique comparator | avoids double-processing and races | correctness + halves comparator launches |
| Shared-memory step fusion | removes per-step global round trips and launches | primary driver: kernel time 80.9 → 64.0 ms as TILE grows |
| Grid-stride load/store | decouples TILE from BLOCK_DIM | enables independent tuning of occupancy vs. traffic |
| Position-based direction | branch-free, data-independent ordering | uniform control flow, low divergence |

## 7. Conclusion and Future Work

The two-kernel bitonic design satisfies the functional requirements on every
graded size and demonstrates a clear optimization path from a launch-bound
global baseline to a shared-memory implementation that minimizes global traffic
and kernel launches. Sweeping the tile size on the PACE-ICE H100 confirmed step
fusion as the primary lever: kernel time fell monotonically from 80.9 ms
(`TILE = 1024`) to 64.0 ms (`TILE = 8192`), with meps rising in step
(465 → 506). `TILE = 4096` was selected as the best balance — 67.6 ms kernel time
(full 10/10 Option-2 score), the highest memory throughput of the sweep, and
~91% shared-kernel occupancy without the resident-block cliff that drops
`TILE = 8192` to 46% aggregate occupancy — for an overall grade of
**16.89 / 20**. Achieved Occupancy (54.6%) and Memory Throughput (47.6%) still
trail the 65% / 75% extra-credit thresholds; because the shared kernel already
reaches ~91% occupancy, closing that gap requires restructuring the global-pass
kernel rather than further tuning.

Promising further optimizations include: (1) **dynamic shared memory** with
`cudaFuncSetAttribute` to push `TILE` beyond 8192 ints; (2) **vectorized
`int4` loads/stores** to raise effective memory bandwidth; (3) processing
**multiple elements per thread** to amortize indexing overhead; and (4) an
initial fused "build" kernel that sorts each tile from scratch in shared memory,
collapsing the earliest stages into a single launch.

# Needleman–Wunsch Algorithm Adaptation for Stereo Matching

## Overview

The **Needleman–Wunsch (NW) algorithm** is a dynamic programming approach traditionally used for global sequence alignment in bioinformatics. This project adapts the NW algorithm for **stereo matching** in computer vision, aiming to find correspondences between pixel intensities along epipolar lines from a pair of rectified stereo images. The ultimate goal is to estimate disparity, which is essential for **3D reconstruction**.

## Algorithm Adaptation

### Epipolar Lines as Sequences

In stereo vision, **epipolar lines** are horizontal lines in rectified images where corresponding points lie. Each epipolar line can be treated as a sequence of pixel intensities. The NW algorithm aligns these sequences to find the best correspondence.

### Dynamic Programming Matrix

A 2D DP matrix `M` of size `(rows + 1) x (cols + 1)` is constructed, where:

- `rows = length of left epipolar line`
- `cols = length of right epipolar line`

Each cell `M[i][j]` represents the best alignment score up to the `i-th` pixel in the left line and the `j-th` pixel in the right line.

### Scoring Scheme

- **Match Score (`matchScore`)**: Reward for matching pixels.
- **Mismatch Penalty (`mismatchPenalty`)**: Penalty for mismatching pixels.
- **Gap Penalty (`gapPenalty`)**: Penalty for introducing gaps (occlusions).

### Matrix Initialization

- The first row and first column are initialized based on gap penalties, representing the cost of aligning with gaps.

### Matrix Filling

Each cell `M[i][j]` is filled based on the following recurrence relation:


### Backtracking

Starting from `M[rows][cols]`, backtrack to determine the optimal alignment path. The path determines the disparity between the corresponding pixels.

## Wavefront Parallelization Strategy

### Concept

**Wavefront Parallelization**, also known as **Anti-Diagonal Parallelization**, processes the DP matrix along its anti-diagonals. Cells on the same anti-diagonal can be computed in parallel as their dependencies lie in the previous anti-diagonals.

### Implementation Steps

1. **Divide the Matrix into Anti-Diagonals:**

   - Each anti-diagonal consists of cells where `i + j = constant`.
   - Cells in the same anti-diagonal can be processed concurrently.

2. **Kernel Launch per Anti-Diagonal:**

   - For each anti-diagonal, launch a CUDA kernel where each thread computes one cell.
   - Ensure that a diagonal is fully processed before moving to the next.

3. **Synchronization Between Anti-Diagonals:**

   - After launching a kernel for a diagonal, synchronize to ensure completion before launching the next.

### Advantages

- **High Parallelism**: Maximizes concurrent computations within each diagonal.
- **Simplified Dependencies**: Only need to wait for the completion of the previous diagonal.

### Optimizations:

- **Shared Memory Usage**
- **Memory Coalescing**
- **Minimizing Thread Divergence**

## CUDA Enhancements and Optimizations

To harness the full potential of CUDA and significantly improve the performance of the Needleman–Wunsch stereo matching algorithm, several enhancements and optimizations have been implemented on the CUDA side:

1. **Single Kernel Launch for All Diagonals**
   
   - **Description:** Instead of launching a separate CUDA kernel for each diagonal of the dynamic programming (DP) matrix, a single kernel launch handles all diagonals. This approach drastically reduces kernel launch overhead, especially for large images with numerous diagonals.
   
   - **Benefits:**
     - **Reduced Overhead:** Minimizes the time spent on initiating kernel launches.
     - **Enhanced Parallelism:** Allows better utilization of CUDA cores by distributing work across threads more efficiently.

2. **Shared Memory Caching**
   
   - **Description:** Utilizes shared memory to cache frequently accessed scanline data (`d_left` and `d_right`). By loading these values into shared memory, the kernel reduces the number of global memory accesses, which are slower compared to shared memory.
   
   - **Benefits:**
     - **Lower Latency:** Shared memory provides faster access times, speeding up data retrieval.
     - **Increased Throughput:** Reduces the bandwidth demand on global memory, allowing for higher data throughput.

3. **Optimized Memory Access Patterns**
   
   - **Description:** Ensures that global memory accesses are coalesced, meaning that consecutive threads access consecutive memory addresses. This optimization maximizes memory throughput and reduces memory transaction overhead.
   
   - **Benefits:**
     - **Improved Memory Bandwidth Utilization:** Maximizes the efficiency of data transfers between global memory and CUDA cores.
     - **Reduced Memory Latency:** Enhances the speed of memory operations, contributing to overall performance gains.

4. **Loop Unrolling and Inlining**
   
   - **Description:** Applies loop unrolling techniques within the CUDA kernel to decrease loop control overhead and increase instruction-level parallelism. Additionally, small functions are inlined to eliminate function call overhead.
   
   - **Benefits:**
     - **Higher Instruction Throughput:** Allows the compiler to better optimize instruction scheduling and parallel execution.
     - **Reduced Overhead:** Lowers the number of instructions related to loop management and function calls.

5. **CUDA Streams for Concurrent Operations**
   
   - **Description:** Implements CUDA streams to overlap data transfers between the host (CPU) and device (GPU) with kernel executions. This concurrency ensures that the GPU remains busy performing computations while simultaneously handling data transfers.
   
   - **Benefits:**
     - **Increased GPU Utilization:** Keeps the GPU occupied, reducing idle times.
     - **Enhanced Throughput:** Overlapping operations lead to more efficient execution and faster overall processing times.

6. **Tiling Techniques for Enhanced Parallelism**
   
   - **Description:** Divides the DP matrix into smaller tiles that can be processed independently by different thread blocks. This division allows multiple tiles to be computed in parallel, maximizing the use of available CUDA cores.
   
   - **Benefits:**
     - **Maximized Parallelism:** Facilitates the concurrent processing of multiple matrix regions.
     - **Scalability:** Improves performance scalability with larger images by efficiently distributing work.

7. **Persistent Memory Allocation**
   
   - **Description:** Allocates device memory once and reuses it across multiple kernel launches or iterations. This strategy avoids the overhead of repeated memory allocations and deallocations.
   
   - **Benefits:**
     - **Reduced Overhead:** Lowers the time spent on memory management operations.
     - **Improved Performance:** Enhances data access speed by maintaining data residency on the GPU.
     
## Disparity Estimation

The disparity is estimated based on the alignment path derived from backtracking the DP matrix. Each step in the path corresponds to a potential disparity between the left and right pixels.




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

### Considerations

- **Load Balancing**: Varying numbers of cells per diagonal can lead to uneven workloads.
- **Memory Access Patterns**: Optimize for coalesced memory accesses to enhance performance.

## Disparity Estimation

The disparity is estimated based on the alignment path derived from backtracking the DP matrix. Each step in the path corresponds to a potential disparity between the left and right pixels.

## Conclusion

Adapting the NW algorithm for stereo matching and implementing it with wavefront parallelization in CUDA provides an efficient way to estimate disparities for 3D reconstruction. This approach leverages GPU parallelism to handle large image datasets effectively.


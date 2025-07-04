# NW Stereo Matching

> **Note**: This project is still a **work in progress**. The current code may not yet provide robust or fully optimized results for all stereo datasets.

An implementation of stereo matching using an adaptation of the Needleman–Wunsch algorithm, with both CPU and CUDA-based parallel versions. This project aims to compute disparity maps between stereo image pairs and compare the results with ground truth data to validate accuracy.

...

## Introduction

Stereo matching is a fundamental problem in computer vision that involves estimating depth by finding correspondences between two images taken from slightly different viewpoints (typically a stereo camera setup). The result is a disparity map, where each pixel value represents the displacement between corresponding pixels in the left and right images.

This project implements a stereo matching algorithm inspired by the Needleman–Wunsch algorithm, commonly used for sequence alignment in bioinformatics. By adapting this algorithm, we aim to compute disparity maps and leverage CUDA for parallel processing to improve performance.

## Background

### Stereo Matching

Stereo matching seeks to determine the depth of objects in a scene by matching pixels from the left and right images. The key challenge is accurately finding corresponding pixels between images, accounting for occlusions, varying textures, and image noise.

**Common Methods in Stereo Matching:**

1. **Local Methods:**
   - **Block Matching (BM)**
   - **Adaptive Support Weight Methods**
   - **Census Transform**

2. **Global Methods:**
   - **Dynamic Programming (DP)**
   - **Graph Cuts**
   - **Belief Propagation**
   - **Semi-Global Matching (SGM)**


### Needleman–Wunsch Algorithm in Stereo Matching

The Needleman–Wunsch algorithm is a dynamic programming method for sequence alignment. In this project, we adapt it for stereo matching by treating each scanline (row of pixels) as sequences to be aligned. The algorithm computes an optimal alignment by maximizing a similarity score, considering match rewards and penalties for mismatches and gaps.

**Benefits:**

- **Global Optimization:** Considers entire scanlines for alignment.
- **Occlusion Handling:** Naturally handles occlusions through gap penalties.

**Challenges:**

- **Computational Complexity:** Standard algorithm has \( O(N^2) \) time and space complexity.
- **Independence of Scanlines:** Processing scanlines independently may ignore vertical coherence.

## CUDA Parallelization

Implementing the Needleman–Wunsch algorithm on CUDA involves parallelizing the dynamic programming matrix computation. We utilize wavefront parallelization, where anti-diagonals of the matrix are computed in parallel, respecting data dependencies.



## Building the Project

### Prerequisites

- **Operating System:** Windows or Linux
- **Compiler:** C++17 compatible compiler (e.g., MSVC, GCC)
- **CUDA Toolkit:** Version 11.7 or compatible
- **CMake:** Version 3.20 or higher
- **OpenCV:** Version 4.8.0 or compatible (with CUDA support if available)
- **Google Test:** For unit testing (fetched automatically via CMake)

### Build Instructions

1. **Clone the Repository:**


   ```bash
   git clone https://github.com/joehrz/nw-stereo-matching.git
   cd nw-stereo-matching
   mkdir build
    cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release
    cmake --build . --config Release
    ./stereo_matching [strategy] left_image.png right_image.png ground_truth.png
    
## Future Work and Optimization Roadmap

### Short-Term Goals
- **Kernel Fusion:**  
  Reduce kernel launch overhead by fusing multiple diagonals into a single kernel launch.
- **Concurrent Processing:**  
  Refactor scanline processing to leverage CUDA streams for concurrent execution of independent scanlines.

### Long-Term Goals
- **Shared Memory Tiling:**  
  Investigate using shared memory to tile large DP matrices, thereby reducing global memory latency.
- **Profiling and Optimization:**  
  Conduct extensive profiling and iterative optimization to fine-tune thread configurations and memory access patterns.

---



   
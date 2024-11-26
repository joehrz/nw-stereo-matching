# NW Stereo Matching

An implementation of stereo matching using an adaptation of the Needleman–Wunsch algorithm, with both CPU and CUDA-based parallel versions. This project aims to compute disparity maps between stereo image pairs and compare the results with ground truth data to validate accuracy.

## Table of Contents

- [Introduction](#introduction)
- [Background](#background)
  - [Stereo Matching](#stereo-matching)
  - [Needleman–Wunsch Algorithm in Stereo Matching](#needleman–wunsch-algorithm-in-stereo-matching)
  - [CUDA Parallelization](#cuda-parallelization)
- [Project Structure](#project-structure)
- [Building the Project](#building-the-project)
  - [Prerequisites](#prerequisites)
  - [Build Instructions](#build-instructions)
- [Running the Program](#running-the-program)
- [Testing and Validation](#testing-and-validation)
- [Results](#results)
- [Future Work](#future-work)
- [References](#references)

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

3. **Deep Learning Methods:**
   - **Convolutional Neural Networks (CNNs)**
   - **Residual Networks and Transformers**

### Needleman–Wunsch Algorithm in Stereo Matching

The Needleman–Wunsch algorithm is a dynamic programming method for sequence alignment. In this project, we adapt it for stereo matching by treating each scanline (row of pixels) as sequences to be aligned. The algorithm computes an optimal alignment by maximizing a similarity score, considering match rewards and penalties for mismatches and gaps.

**Benefits:**

- **Global Optimization:** Considers entire scanlines for alignment.
- **Occlusion Handling:** Naturally handles occlusions through gap penalties.

**Challenges:**

- **Computational Complexity:** Standard algorithm has \( O(N^2) \) time and space complexity.
- **Independence of Scanlines:** Processing scanlines independently may ignore vertical coherence.

### CUDA Parallelization

Implementing the Needleman–Wunsch algorithm on CUDA involves parallelizing the dynamic programming matrix computation. We utilize wavefront parallelization, where anti-diagonals of the matrix are computed in parallel, respecting data dependencies.

**Optimizations:**

- **Shared Memory Usage**
- **Memory Coalescing**
- **Minimizing Thread Divergence**

## Project Structure
```bash
stereo-matching-needleman/
│
├── README.md                           # Project description and setup instructions
├── CMakeLists.txt                      # CMake configuration for building the project 
├── src/                                # Source files
|   ├── main.cpp                                   
│   ├── cpu/                            # Serial version of the stereo matching algorithm  
|   |   └── stereo_matching.cpp
│   │    
│   ├── cuda/                           # CUDA version 
│   │   ├── stereo_matching_cuda.cu                
│   │   ├── stereo_matching_kernel.cu
|   |   └── stereo_matching_kernel.h
|   | 
|
├── include/
│   └── stereo_matching.h           
├── data/
│   ├── cone/                    # Folder for stereo images used in testing
│       ├── im2.png                 # Sample left stereo image
│       └── im6.png                 # Sample right stereo image
│       └── disp2.png                 # Sample right stereo image
|
├── tests/
│   ├── CMakeLists.txt 
│   └──unit
|       └── test_stereo_matching.cpp
└── docs/                       # Documentation for the project
    └── algorithm.md            # Detailed explanation of the modified Needleman-Wunsch algorithm for stereo matching


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
   git clone https://github.com/yourusername/nw-stereo-matching.git
   cd nw-stereo-matching
   mkdir build
    cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release
    cmake --build . --config Release
    ./stereo_matching [strategy] left_image.png right_image.png ground_truth.png
   
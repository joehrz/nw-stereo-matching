# Stereo Matching with Needleman–Wunsch Algorithm

This project extends the **Needleman–Wunsch algorithm**, traditionally used for sequence alignment, to the domain of **computer vision**. The algorithm is adapted for **stereo matching**, where the goal is to find correspondences between pixel intensities along epipolar lines from a pair of stereo images. Ultimately, this can be used as part of a pipeline for **3D reconstruction**.

## Features
- **Global alignment** of pixel intensities between two scan lines (epipolar lines) from rectified stereo images.
- Incorporates penalties for **gaps** (occlusions) in the image matching process, similar to handling insertions/deletions in sequence alignment.
- Scalable design for future **GPU parallelization** using CUDA.

## Project Goals
- Implement a **C++ version** of the Needleman–Wunsch algorithm tailored for stereo matching.
- Extend the algorithm to run on the **GPU using CUDA** for efficient performance on large image datasets.
- Create a robust stereo matching system that handles occlusions and disparity estimation for 3D reconstruction.

## Repository Structure
```bash
stereo-matching-needleman/
│
├── README.md                # Project description and setup instructions
├── src/                     # Source files
│   ├── nw_stereo_matching.cpp  # Serial version of the stereo matching algorithm
│   ├── nw_stereo_matching.cu   # CUDA version (to be added later)
│   └── CMakeLists.txt        # CMake configuration for building the project (optional)
├── images/                  # Folder for stereo images used in testing
│   └── left_image.png        # Sample left stereo image
│   └── right_image.png       # Sample right stereo image
└── docs/                    # Documentation for the project
    └── algorithm.md          # Detailed explanation of the modified Needleman-Wunsch algorithm for stereo matching

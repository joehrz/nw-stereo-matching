#include <iostream>
#include <numeric> // For std::accumulate
#include <opencv2/opencv.hpp>
#include "stereo_matching.h"
#include <chrono> // For timing
#include <vector>
#include <omp.h>   // For OpenMP parallelization
#include <cmath>   // For std::round, std::abs, std::max, std::min, std::sqrt
#include <algorithm> // For std::min, std::max

int main(int argc, char** argv) {
    if(argc != 5) {
        std::cerr << "Usage: ./stereo_matching [strategy] left_image.png right_image.png ground_truth.png" << std::endl;
        std::cerr << "Strategies: cpu, cuda_wavefront, cuda_wavefront_fused" << std::endl;
        return -1;
    }

    std::string strategy = argv[1];
    std::string leftPath = argv[2];
    std::string rightPath = argv[3];
    std::string gtPath = argv[4];

    // Load images in grayscale
    cv::Mat leftImg = cv::imread(leftPath, cv::IMREAD_GRAYSCALE);
    cv::Mat rightImg = cv::imread(rightPath, cv::IMREAD_GRAYSCALE);

    if(leftImg.empty()) {
        std::cerr << "Error loading left image: " << leftPath << std::endl;
        
        return -1;
    }
    if(rightImg.empty()) {
        std::cerr << "Error loading right image: " << rightPath << std::endl;
        return -1;
    }

    // Ensure images have the same number of rows (epipolar lines)
    if(leftImg.rows != rightImg.rows) {
        std::cerr << "Left and right images must have the same number of rows (epipolar lines)." << std::endl;
        return -1;
    }

    // Load ground truth disparity map
    cv::Mat gtDisparity = cv::imread(gtPath, cv::IMREAD_UNCHANGED); // Use IMREAD_UNCHANGED to preserve precision
    if(gtDisparity.empty()) {
        std::cerr << "Error loading ground truth disparity map: " << gtPath << std::endl;
        return -1;
    }

    // Adjust for scaling factor (assuming GT stores d*256)
    double scalingFactor = 1.0 / 256.0; 
    cv::Mat gtDisparityFloat;
    gtDisparity.convertTo(gtDisparityFloat, CV_32F, scalingFactor);

    // Verify ground truth disparity statistics
    double minValGT, maxValGT; // Renamed to avoid conflict
    cv::minMaxLoc(gtDisparityFloat, &minValGT, &maxValGT);
    std::cout << "Ground Truth Disparity - Min: " << minValGT << ", Max: " << maxValGT << std::endl;

    int totalRows = leftImg.rows;
    int totalCols = leftImg.cols;

    // Initialize disparity map for integer results
    cv::Mat disparityMap(totalRows, totalCols, CV_32F, cv::Scalar(-1.0f));

    // Initialize StereoMatcher with scoring parameters
    // Note: matchScore and mismatchPenalty are not directly used by the current cost function in fillMatrix/kernels
    // int matchScore = 12;     
    // int mismatchPenalty = -3; 
    // int gapPenalty = -5;      

    // StereoMatcher matcher(matchScore, mismatchPenalty, gapPenalty);


    // Option 2: Emphasize matches, stronger mismatch penalty, very severe gap penalty
    int matchScore = 5;          // Very high reward for perfect match
    int mismatchPenalty = 0;    // Very high base penalty for any mismatch
    int gapPenalty = -1;        // Extremely costly gaps

    // Option 3: Focus on minimizing differences, moderate gap penalty
    // int matchScore = 0;      // No explicit reward for perfect match beyond zero difference cost
    // int mismatchPenalty = 0;   // Base mismatch penalty is 0, cost only from truncatedDiff
    // int gapPenalty = -10;    

    std::cout << "\nUsing Scoring: matchScore=" << matchScore 
              << ", mismatchPenalty=" << mismatchPenalty
              << ", gapPenalty=" << gapPenalty << std::endl;

    StereoMatcher matcher(matchScore, mismatchPenalty, gapPenalty);










    std::cout << "\nStarting Needleman-Wunsch alignment (" << strategy << ")..." << std::endl;
    auto startNW = std::chrono::high_resolution_clock::now();

    // Parallel processing using OpenMP for scanlines
    #pragma omp parallel for schedule(dynamic)
    for(int row = 0; row < totalRows; ++row) {
        std::vector<int> leftLine(totalCols);
        std::vector<int> rightLine(totalCols);
        for(int col = 0; col < totalCols; ++col) {
            leftLine[col] = static_cast<int>(leftImg.at<uchar>(row, col));
        }
        for(int col = 0; col < totalCols; ++col) {
            rightLine[col] = static_cast<int>(rightImg.at<uchar>(row, col));
        }

        AlignmentResult result;
        if(strategy == "cpu") {
            result = matcher.computeAlignment(leftLine, rightLine);
        } else if(strategy == "cuda_wavefront") {
            result = matcher.computeAlignmentCUDA(leftLine, rightLine);
        } else if(strategy == "cuda_wavefront_fused") {
            result = matcher.computeAlignmentCUDA_Fused(leftLine, rightLine);
        } else {
            #pragma omp critical
            {
                if (row == 0) { // Print error only once
                    std::cerr << "Unknown strategy: " << strategy << std::endl;
                    std::cerr << "Available strategies: cpu, cuda_wavefront, cuda_wavefront_fused" << std::endl;
                }
            }
            // To prevent proceeding with an uninitialized 'result', fill with -1
            for(int col = 0; col < totalCols; ++col) {
                disparityMap.at<float>(row, col) = -1.0f;
            }
            continue; 
        }

        std::vector<float> estimatedDisparities(totalCols, -1.0f);
        size_t idxLeftPos = 0;   
        size_t idxRightPos = 0;  

        for(size_t idx = 0; idx < result.leftAligned.size(); ++idx) {
            if(result.leftAligned[idx] != -1 && result.rightAligned[idx] != -1) {
                int disparity_val = static_cast<int>(idxLeftPos) - static_cast<int>(idxRightPos);
                if(idxLeftPos < estimatedDisparities.size()) {
                    estimatedDisparities[idxLeftPos] = static_cast<float>(disparity_val);
                }
                idxLeftPos++;
                idxRightPos++;
            } else {
                if(result.leftAligned[idx] != -1) idxLeftPos++;
                if(result.rightAligned[idx] != -1) idxRightPos++;
            }
        }

        for(int col = 0; col < totalCols; ++col) {
            disparityMap.at<float>(row, col) = estimatedDisparities[col];
        }
    }

    auto endNW = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> alignmentTime = endNW - startNW;
    std::cout << "Alignment Time (" << strategy << "): " << alignmentTime.count() << " ms" << std::endl;

    // --- Metrics for Integer Disparity Map ---
    std::cout << "\n--- Integer Disparity Results ---" << std::endl;
    double totalErrorNW_int = 0.0;
    int validPixelCountNW_int = 0;
    int badPixelCountNW_int = 0;
    double pbpThreshold = 1.0; 

    for(int row = 0; row < totalRows; ++row) {
        for(int col = 0; col < totalCols; ++col) {
            float estDisp = disparityMap.at<float>(row, col); // Raw integer disparity
            float gtDisp = gtDisparityFloat.at<float>(row, col); // GT scaled to pixel units

            if(estDisp >= 0 && gtDisp > 0) { // Valid disparities for comparison
                float error = std::abs(estDisp - gtDisp);
                totalErrorNW_int += error;
                validPixelCountNW_int++;
                if(error > pbpThreshold) {
                    badPixelCountNW_int++;
                }
            }
        }
    }

    double MAE_int = (validPixelCountNW_int > 0) ? (totalErrorNW_int / validPixelCountNW_int) : 0.0;
    double RMSE_int = 0.0;
    if(validPixelCountNW_int > 0) {
        double totalSqError_int = 0.0;
        for(int row = 0; row < totalRows; ++row) {
            for(int col = 0; col < totalCols; ++col) {
                float estDisp = disparityMap.at<float>(row, col); // Raw integer disparity
                float gtDisp = gtDisparityFloat.at<float>(row, col);
                if(estDisp >= 0 && gtDisp > 0) {
                    double error = estDisp - gtDisp;
                    totalSqError_int += error * error;
                }
            }
        }
        RMSE_int = std::sqrt(totalSqError_int / validPixelCountNW_int);
    }
    double PBP_int = (validPixelCountNW_int > 0) ? ((double)badPixelCountNW_int / validPixelCountNW_int) * 100.0 : 0.0;

    double sumEstDisparityNW_int = 0.0;
    int estValidCountNW_int = 0;
    double sumGtDisparity = 0.0; // GT sum is independent of estimation
    int gtValidCount = 0;

    for(int row = 0; row < totalRows; ++row) {
        for(int col = 0; col < totalCols; ++col) {
            float estDisp = disparityMap.at<float>(row, col);
            float gtDisp = gtDisparityFloat.at<float>(row, col);
            if(estDisp >= 0) {
                sumEstDisparityNW_int += estDisp;
                estValidCountNW_int++;
            }
            if(gtDisp > 0) {
                sumGtDisparity += gtDisp;
                gtValidCount++;
            }
        }
    }
    double meanEstDisparityNW_int = (estValidCountNW_int > 0) ? (sumEstDisparityNW_int / estValidCountNW_int) : 0.0;
    double meanGtDisparity = (gtValidCount > 0) ? (sumGtDisparity / gtValidCount) : 0.0; // Calculated once

    std::cout << "Mean Estimated Disparity (Integer NW): " << meanEstDisparityNW_int << std::endl;
    std::cout << "Ground Truth Mean Disparity: " << meanGtDisparity << std::endl;
    std::cout << "Mean Absolute Error (MAE) (Integer NW): " << MAE_int << std::endl;
    std::cout << "Root Mean Squared Error (RMSE) (Integer NW): " << RMSE_int << std::endl;
    std::cout << "Percentage of Bad Pixels (PBP) (Integer NW): " << PBP_int << "%" << std::endl;

    // --- Visualize Integer Disparity Map (Properly Scaled) ---
    cv::Mat validMask_int = (disparityMap >= 0); 
    double computedMin_int, computedMax_int;
    cv::minMaxLoc(disparityMap, &computedMin_int, &computedMax_int, nullptr, nullptr, validMask_int);

    cv::Mat disparityMapVis_int;
    if (computedMax_int > computedMin_int) { 
        cv::Mat toVisualize_int = disparityMap.clone(); 
        for (int r = 0; r < toVisualize_int.rows; ++r) {
            for (int c = 0; c < toVisualize_int.cols; ++c) {
                if (toVisualize_int.at<float>(r, c) < 0) {
                    toVisualize_int.at<float>(r, c) = static_cast<float>(computedMin_int);
                }
            }
        }
        cv::minMaxLoc(toVisualize_int, &computedMin_int, &computedMax_int, nullptr, nullptr, validMask_int); 
        if (computedMax_int > computedMin_int) { 
             toVisualize_int.convertTo(disparityMapVis_int, CV_8U, 255.0 / (computedMax_int - computedMin_int), -computedMin_int * 255.0 / (computedMax_int - computedMin_int));
        } else { 
            toVisualize_int.convertTo(disparityMapVis_int, CV_8U, 0, 255); 
        }
    } else {
        disparityMap.convertTo(disparityMapVis_int, CV_8U); 
    }
    cv::applyColorMap(disparityMapVis_int, disparityMapVis_int, cv::COLORMAP_JET);
    cv::imwrite("DisparityMap_NW_IntegerScaled.png", disparityMapVis_int);
    std::cout << "Disparity Map (Integer Scaled) saved to: DisparityMap_NW_IntegerScaled.png" << std::endl;


    // --- BEGIN SUB-PIXEL REFINEMENT ---
    cv::Mat refinedDisparityMap = disparityMap.clone(); 

    std::cout << "\nStarting sub-pixel refinement..." << std::endl;
    auto startSubPixelRefinement = std::chrono::high_resolution_clock::now();

    #pragma omp parallel for schedule(dynamic)
    for (int r = 0; r < totalRows; ++r) {
        for (int c = 0; c < totalCols; ++c) {
            float d_int_float = disparityMap.at<float>(r, c);
            if (d_int_float < 0) { 
                refinedDisparityMap.at<float>(r, c) = d_int_float; 
                continue;
            }
            int d_int = static_cast<int>(std::round(d_int_float));

            auto calculate_matching_cost = [&](int left_image_col_idx, int disparity_val) {
                int right_image_col_idx = left_image_col_idx - disparity_val;
                if (right_image_col_idx >= 0 && right_image_col_idx < totalCols) {
                    int intensity_diff = std::abs(
                        static_cast<int>(leftImg.at<uchar>(r, left_image_col_idx)) -
                        static_cast<int>(rightImg.at<uchar>(r, right_image_col_idx))
                    );
                    return static_cast<float>(std::min(intensity_diff, 20));
                }
                return 21.0f; 
            };

            float C0 = calculate_matching_cost(c, d_int);        
            float C_minus1 = calculate_matching_cost(c, d_int - 1); 
            float C_plus1 = calculate_matching_cost(c, d_int + 1);  

            if (C0 <= 20.0f && C_minus1 <= 20.0f && C_plus1 <= 20.0f && C0 <= C_minus1 && C0 <= C_plus1) {
                float denominator = C_minus1 - 2.0f * C0 + C_plus1;
                if (std::abs(denominator) > 1e-5) { 
                    float delta = (C_minus1 - C_plus1) / (2.0f * denominator);
                    delta = std::max(-0.5f, std::min(0.5f, delta));
                    refinedDisparityMap.at<float>(r, c) = static_cast<float>(d_int) + delta;
                }
            }
        }
    }
    auto endSubPixelRefinement = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> subPixelTime = endSubPixelRefinement - startSubPixelRefinement;
    std::cout << "Sub-pixel refinement time: " << subPixelTime.count() << " ms" << std::endl;
    // --- END SUB-PIXEL REFINEMENT ---

    // --- Metrics for Refined Disparity Map ---
    std::cout << "\n--- Refined Disparity Results ---" << std::endl;
    double totalErrorNW_refined = 0.0;
    int validPixelCountNW_refined = 0; 
    int badPixelCountNW_refined = 0;
    // pbpThreshold is already defined (e.g., 1.0)

    for(int row = 0; row < totalRows; ++row) {
        for(int col = 0; col < totalCols; ++col) {
            float estDisp = refinedDisparityMap.at<float>(row, col); 
            float gtDisp = gtDisparityFloat.at<float>(row, col);    

            if(estDisp >= 0 && gtDisp > 0) { 
                float error = std::abs(estDisp - gtDisp);
                totalErrorNW_refined += error;
                validPixelCountNW_refined++;
                if(error > pbpThreshold) {
                    badPixelCountNW_refined++;
                }
            }
        }
    }

    double MAE_refined = (validPixelCountNW_refined > 0) ? (totalErrorNW_refined / validPixelCountNW_refined) : 0.0;
    double RMSE_refined = 0.0;
    if(validPixelCountNW_refined > 0) {
        double totalSqError_refined = 0.0;
        for(int row = 0; row < totalRows; ++row) {
            for(int col = 0; col < totalCols; ++col) {
                float estDisp = refinedDisparityMap.at<float>(row, col); 
                float gtDisp = gtDisparityFloat.at<float>(row, col);
                if(estDisp >= 0 && gtDisp > 0) {
                    double error = estDisp - gtDisp;
                    totalSqError_refined += error * error;
                }
            }
        }
        RMSE_refined = std::sqrt(totalSqError_refined / validPixelCountNW_refined);
    }
    double PBP_refined = (validPixelCountNW_refined > 0) ? ((double)badPixelCountNW_refined / validPixelCountNW_refined) * 100.0 : 0.0;
    
    double sumEstDisparityNW_refined = 0.0;
    int estValidCountNW_refined = 0;
    for(int row = 0; row < totalRows; ++row) {
        for(int col = 0; col < totalCols; ++col) {
            float estDisp = refinedDisparityMap.at<float>(row, col); 
            if(estDisp >= 0) { 
                sumEstDisparityNW_refined += estDisp;
                estValidCountNW_refined++;
            }
        }
    }
    double meanEstDisparityNW_refined = (estValidCountNW_refined > 0) ? (sumEstDisparityNW_refined / estValidCountNW_refined) : 0.0;

    std::cout << "Mean Estimated Disparity (Refined NW): " << meanEstDisparityNW_refined << std::endl;
    std::cout << "Ground Truth Mean Disparity: " << meanGtDisparity << std::endl; 
    std::cout << "Mean Absolute Error (MAE) (Refined NW): " << MAE_refined << std::endl;
    std::cout << "Root Mean Squared Error (RMSE) (Refined NW): " << RMSE_refined << std::endl;
    std::cout << "Percentage of Bad Pixels (PBP) (Refined NW): " << PBP_refined << "%" << std::endl;

    // --- Visualize Refined Disparity Map ---
    cv::Mat validMask_refined = (refinedDisparityMap >= 0); 
    double computedMin_refined, computedMax_refined;
    cv::minMaxLoc(refinedDisparityMap, &computedMin_refined, &computedMax_refined, nullptr, nullptr, validMask_refined);

    cv::Mat disparityMapVis_refined;
    if (computedMax_refined > computedMin_refined) {
        cv::Mat toVisualize_refined = refinedDisparityMap.clone();
        for (int r = 0; r < toVisualize_refined.rows; ++r) {
            for (int c = 0; c < toVisualize_refined.cols; ++c) {
                if (toVisualize_refined.at<float>(r, c) < 0) {
                    toVisualize_refined.at<float>(r, c) = static_cast<float>(computedMin_refined);
                }
            }
        }
        // Optional: Re-evaluate min/max if -1s were changed and you want them excluded from normalization range
        // cv::minMaxLoc(toVisualize_refined, &computedMin_refined, &computedMax_refined, nullptr, nullptr, validMask_refined); 
        
        if (computedMax_refined > computedMin_refined) { 
             toVisualize_refined.convertTo(disparityMapVis_refined, CV_8U, 255.0 / (computedMax_refined - computedMin_refined), -computedMin_refined * 255.0 / (computedMax_refined - computedMin_refined));
        } else { 
            toVisualize_refined.convertTo(disparityMapVis_refined, CV_8U, 0, 255); 
        }
    } else {
        refinedDisparityMap.convertTo(disparityMapVis_refined, CV_8U); 
    }
    cv::applyColorMap(disparityMapVis_refined, disparityMapVis_refined, cv::COLORMAP_JET);
    cv::imwrite("DisparityMap_NW_SubPixel.png", disparityMapVis_refined); 
    std::cout << "Disparity Map (Sub-Pixel Refined) saved to: DisparityMap_NW_SubPixel.png" << std::endl;

    // Print some rows of initial integer disparities
    std::cout << "\n--- Sample Initial Integer Disparities ---" << std::endl;
    for (int row = 0; row < 5 && row < totalRows; ++row) {
        std::cout << "Row " << row << " disparities: ";
        for (int col = 0; col < totalCols; ++col) {
            std::cout << disparityMap.at<float>(row, col) << " ";
        }
        std::cout << std::endl;
    }
    // Print some rows of refined disparities
    std::cout << "\n--- Sample Refined Sub-Pixel Disparities ---" << std::endl;
    for (int row = 0; row < 5 && row < totalRows; ++row) {
        std::cout << "Refined Row " << row << " disparities: ";
        for (int col = 0; col < totalCols; ++col) {
            std::cout << refinedDisparityMap.at<float>(row, col) << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}
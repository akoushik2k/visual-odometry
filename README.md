# Visual Odometry

## Overview
This repository contains the implementation of a stereo visual odometry algorithm that estimates the position of objects by analyzing frame-to-frame motion from successive stereo images. The method operates without prior assumptions about camera motion and utilizes dense disparity images from stereo cameras. The algorithm has been tested on the KITTI dataset and further evaluated using real-world data from a ZED stereo camera.

## Features
- **Stereo Visual Odometry**: Utilizes stereo images for better depth estimation.
- **Feature-Based Pose Estimation**: Implements SIFT and BFMatcher for feature detection and matching.
- **Disparity and Depth Calculation**: Computes disparity maps using StereoSGBM and extracts depth information.
- **Real-Time Pose Estimation**: Uses solvePnPRansac for estimating camera motion.
- **KITTI Dataset Benchmarking**: Evaluates algorithm performance using standard KITTI sequences.
- **Custom Dataset Integration**: Supports data collection and processing from a ZED stereo camera.

## Dataset
The model is trained and evaluated on:
1. **KITTI Visual Odometry Benchmark**: A dataset for autonomous vehicle navigation tasks.
2. **ZED Stereo Camera Data**: Custom-collected stereo images used for real-world validation.

## Methodology
### 1. Data Preparation
- Extract and rectify stereo images.
- Compute disparity maps using StereoSGBM.
- Generate depth maps from disparity information.

### 2. Feature Extraction & Matching
- Extract keypoints using **SIFT**.
- Match keypoints across frames with **BFMatcher**.
- Filter matches using Lowe's ratio test.

### 3. Pose Estimation
- Compute **Fundamental Matrix** for epipolar geometry.
- Derive **Essential Matrix** using camera intrinsics.
- Decompose the **Essential Matrix** to obtain **rotation and translation**.
- Use **solvePnPRansac** for accurate pose estimation.

### 4. Trajectory Reconstruction
- Compute the homogeneous transformation matrix.
- Track vehicle motion over time and visualize results.

# Fall Detection System - Comprehensive Technical Documentation

## 1. Project Overview & Architectural Vision

This Documentation Provides An In-Depth Technical Analysis Of The **CV (Computer Vision)** Repository, A Specialized Fall Detection System Designed For Real-Time Surveillance & Safety Monitoring. The Core Application Leverages State-Of-The-Art Pose Estimation Models To Analyze Human Movement Patterns & Detect Anomalous Events Such As Falls.

The System Is Built Upon **Python** & Integrates **Ultralytics YOLOv11** For Robust Keypoint Extraction, Optimized With **OpenVINO** For High-Performance Inference On Intel Hardware. It Features A Multi-Threaded Architecture To Decouple Inference From Video Rendering, Ensuring Smooth Playback Even Under Computational Load. Beyond Detection, The System Includes Automated Event Logging, Video Clippings Of Incidents, & Audio Alerts, Making It A Complete End-To-End Solution For Automated Monitoring.

From An Architectural Perspective, The Project Is Structured Around A Main Execution Loop That Orchestrates Video Capture, Asynchronous Inference, Heuristic-Based Logic Analysis, & I/O Operations (Disk Writing & Audio Playback).

---

## 2. Technology Stack & Infrastructure Decisions

### 2.1 Core Technologies

The Application Is Constructed Using High-Performance Computer Vision Libraries & Frameworks.

* **Language**: **Python 3.13**
    * **Rationale**: Selected For Its Extensive Ecosystem Of Data Science & Computer Vision Libraries.
* **Computer Vision**: **OpenCV (cv2)**
    * **Role**: Handles Video I/O, Image Manipulation (Resizing, Drawing), & Window Management. It Is The Backbone Of The Visual Pipeline.
* **Model Framework**: **Ultralytics YOLO (v11 Pose)**
    * **Role**: Performs Pose Estimation To Extract Skeletal Keypoints (Shoulders, Hips, Etc.) From Video Frames. The `yolo11n-pose.pt` Model Is Used For Its Balance Between Speed & Accuracy.
* **Inference Optimization**: **OpenVINO Toolkit**
    * **Implementation**: The System Compiles The YOLO Model Into An OpenVINO Intermediate Representation (IR) To Accelerate Inference On Intel CPUs & GPUs. It Manages Thread Counts (`OV_CPU_THREADS_NUM`) & Device Compilation Explicitly.
* **Concurrency**: **Python ThreadPoolExecutor**
    * **Architecture**: Inference Tasks Are Offloaded To A Worker Thread Pool. This Non-Blocking Approach Prevents The Heavy Model Prediction Step From Freezing The GUI Or Video Feed.

### 2.2 Infrastructure & Environment

The Project Utilizes **Conda** For Environment Management, Ensuring Reproducible Dependency Chains.

* **Dependencies**:
    * `opencv`, `ultralytics`, `openvino`, `ffmpeg`: Core Vision & AI Libraries.
    * `torch`, `torchvision`: Backend Deep Learning Frameworks Required By YOLO.
* **Model Export**:
    * The Setup Script Includes Commands To Export The PyTorch Model (`.pt`) To OpenVINO Format For Production Deployment.

---

## 3. Detailed System Architecture & Logic

### 3.1 Fall Detection Algorithm

The Core Intelligence Lies In The Heuristic Analysis Of Skeletal Keypoints Extracted By The YOLO Model.

* **Keypoint Extraction**:
    * The System Tracks Specific Keypoints: Shoulders (Indices 5, 6) & Hips (Indices 11, 12).
* **Geometric Metrics**:
    1. **Torso Angle**: The Angle Of The Vector Connecting The Mid-Shoulder To Mid-Hip Relative To The Vertical Axis. A High Angle (> 45°) Suggests The Person Is Leaning Or Lying Down.
    2. **Height Ratio**: The Ratio Of The Vertical Distance Between Shoulders & Hips To The Actual Torso Length. A Low Ratio (< 0.3) Indicates Vertical Compression (Crumpling).
    3. **Hips Ratio**: The Normalized Y-Position Of The Hips Relative To The Frame Height. A High Ratio (> 0.85) Indicates The Person Is Close To The Ground.
* **Velocity Tracking**:
    * The System Maintains A `trackedPersons` Dictionary To Store Previous Positions. It Calculates The Vertical Velocity Of The Torso Over Time. Rapid Downward Movement (> 0.25 Threshold) Serves As An Early Indicator Of A Fall.
* **Detection Triggers**:
    * A Fall Is Confirmed If A Combination Of These Conditions Is Met (E.g., High Angle + Compression, Or High Velocity + Low Height).
* **Fallback Mechanism**:
    * If Pose Keypoints Are Unreliable (Low Confidence), The System Reverts To Bounding Box Analysis. If The Aspect Ratio (Width/Height) Exceeds 1.2, It Is Flagged As A Potential Fall.

### 3.2 Event Handling & I/O

Upon Detecting A Fall, The System Triggers A Cascade Of Response Actions.

* **Visual & Audio Alerts**:
    * **UI Updates**: The Bounding Box Turns Red, & The Label Switches To "Falling".
    * **Sound**: Uses `winsound` To Play An Alert Wav File (`sound.wav`) Asynchronously.
* **Data Logging**:
    * **CSV Log**: Appends A Timestamped Entry To `exp/exp.csv` With Event Type & Coordinates.
    * **Snapshot**: Saves A High-Resolution Image Of The Frame (`.jpg`) To The Experiment Directory.
* **Video Evidence Recording**:
    * **Buffering**: Maintains A Rolling `deque` Buffer Of Previous Frames (Pre-Event Context).
    * **Recording**: When A Fall Occurs, It Writes The Buffered Frames Plus A Configured Duration Of Post-Event Frames To A New Video File (`.mp4`). This Ensures The Context Leading Up To The Fall Is Captured.

---

## 4. Project Directory Structure

The Repository Is Organized To Separate Operational Code From Development Tools.

* **`main.py`**: The Primary Application Entry Point Containing The Inference Loop & Logic.
* **`conda`**: Setup Script Defining Environment Dependencies & Model Export Commands.
* **`dev/`**: A Directory Containing Experimental Scripts Named After Animals (E.g., `Antelope.py`, `Bison.py`).
    * `Antelope.py`: A Lightweight Script For Testing Video Capture Performance & FPS Calculation Without Heavy Inference.
* **`yolo11n-pose_openvino_model/`**: (Generated) Directory Containing The Optimized OpenVINO Model Files (`.xml`, `.bin`).
* **`exp/`**: (Generated) Runtime Directory For Storing Logs (`exp.csv`), Snapshots, & Recorded Videos.
* **`sound.wav`**: Audio Asset Used For Fall Alarms.

---

## 5. Configuration & Customization

The `main.py` Script Includes A Dedication Configuration Section Allow Tuning Of Sensitivity & System Behavior.

### 5.1 Detection Thresholds
* `fallCooldown`: Minimum Time (Seconds) Between Consecutive Fall Alerts (Default: 1.0s).
* `minTrackingConfidence`: Confidence Threshold For Keypoint Validity (Default: 0.4).
* `torsoAngleThreshold`: Angle In Degrees To Determine Leaning (Default: 45).
* `heightRatioThreshold`: Ratio To Determine Vertical Compression (Default: 0.3).
* `velocityThreshold`: Speed Threshold For Dynamic Fall Detection (Default: 0.25).

### 5.2 System Settings
* `videoPath`: Input Source (Default: "0.mp4", Can Be Set To `0` For Webcam).
* `frameWidth`, `frameHeight`: Input Resolution (Default: 640x360).
* `maxDetections`: Maximum Number Of People To Track Simultaneously (Default: 60).
* `drawKeypoints`, `drawSkeleton`: Boolean Flags To Toggle Visualization Layers.

---

## 6. Installation & Setup Guide

### 6.1 Environment Setup
The Project Relies On Conda For Dependency Management.

1.  **Create Environment**:
    Initialize A New Conda Environment With Python 3.13.
    ```bash
    conda create --prefix ./env python=3.13.2
    conda activate ./env
    ```
2.  **Install Dependencies**:
    Install Required Libraries From Conda-Forge & PyPI.
    ```bash
    conda install -c conda-forge opencv ultralytics openvino ffmpeg
    pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cpu](https://download.pytorch.org/whl/cpu)
    ```
3.  **Model Optimization**:
    Export The YOLO Model To OpenVINO Format For Hardware Acceleration.
    ```bash
    yolo export model=yolo11n-pose.pt format=openvino
    ```

### 6.2 Running The Application
1.  **Place Assets**: Ensure `sound.wav` & The Video Source (`0.mp4`) Are In The Root Directory.
2.  **Execute**:
    ```bash
    python main.py
    ```
3.  **Controls**:
    * The Application Window Will Display The Video Feed With Overlays.
    * Press `q` To Terminate The Program Safely.

---

## 7. License

This Project Is A Custom Implementation Using Open-Source Libraries. The Underlying Models (YOLO) & Libraries (OpenCV, OpenVINO) Are Subject To Their Respective Licenses (AGPL-3.0 For Ultralytics, Apache 2.0 For OpenVINO/OpenCV).

# YOLO_Depth

This project integrates YOLO object detection with RealSense RGB-D cameras to detect objects and estimate their 3D positions using depth information. The system publishes the object's position as a `tf` message

## Features
- YOLO-based object detection.
- Real-time 3D position estimation using Intel RealSense cameras.
- Object position broadcasted as a `tf` frame.
- Visualization of detected objects and their bounding boxes.

## Dependencies

### Environment
- Ubuntu 20.04
- ROS Noetic
- Intel RealSense RGB-D camera (e.g., D455)

### Installations
Make sure to install the following dependencies:

1. **ROS Noetic and ROS dependencies:**
   ```bash
   sudo apt-get install ros-noetic-ros-numpy

2. **Python dependencies:**
   ```bash
    pip install python3-pcl
    pip install pyrealsense2
    pip install ultralytics

![image](https://github.com/user-attachments/assets/6ae0d3d5-f682-4271-b6be-dfbd48320d24)


https://github.com/user-attachments/assets/e8c1c63a-7312-494d-8c04-03681b1c9bc2



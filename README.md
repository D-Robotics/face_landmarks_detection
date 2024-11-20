# face_landmarks_detection

English| [简体中文](./README_CN.md)

## Description

Face Landmarks Detection Package.

## Supported Platforms

| Material Name   | Manufacturer | Reference Links                                                                                                                                             |
| --------------- | ------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| RDK X3 / RDK X5 | Multiple     | [RDK X3](https://developer.d-robotics.cc/rdkx3)<br>[RDK X5](https://developer.d-robotics.cc/rdkx5)                                                          |
| Camera          | Multiple     | [MIPI Camera](https://developer.horizon.cc/nodehubdetail/168958376283445781)<br>[USB Camera](https://developer.horizon.cc/nodehubdetail/168958376283445777) |

## Building

- Building depends on [ai_msgs](https://github.com/D-Robotics/hobot_msgs)
- Building depends on [hobot_dnn](https://github.com/D-Robotics/hobot_dnn)

```shell
# Cross-compilation for X5
bash build.sh -p X5 -s img_msgs
bash build.sh -p X5 -s hbm_img_msgs
bash build.sh -p X5 -s ai_msgs
bash build.sh -p X5 -s dnn_node
bash build.sh -p X5 -s face_landmarks_detection
```

## Running Commands

1. Start the body, face, hand detection algorithms, and the face landmarks detection algorithm:

```shell
===============================================================================================================================
# Load the tros.b environment
source /opt/tros/humble/setup.bash
source ./install/local_setup.bash
===============================================================================================================================
# Offline inference
ros2 launch face_landmarks_detection face_landmarks_det_node.launch.py feed_type:=1 feed_image_path:=image.png roi_xyxy:=x1,y1,x2,y2,x3,y3,x4,y4,...

# Example
ros2 launch face_landmarks_detection face_landmarks_det_node.launch.py feed_type:=1 feed_image_path:=image.png roi_xyxy:=251,242,328,337
===============================================================================================================================
# Online inference, requires both the body detection and face landmarks detection algorithm nodes to be started
cp -r /opt/tros/${TROS_DISTRO}/lib/mono2d_body_detection/config/ .

# For USB camera
export CAM_TYPE=usb
# For MIPI camera
export CAM_TYPE=mipi

# Start the launch file
ros2 launch face_landmarks_detection body_det_face_landmarks_det.launch.py
===============================================================================================================================
```

## Running Results

Detects 106 landmarks on the face.

![](./doc/face_landmarks_det_render.png)

## Package Parameters

| Name                  | Default Value                   | Description                                                                                        |
| --------------------- | ------------------------------- | -------------------------------------------------------------------------------------------------- |
| feed_type             | 0                               | 0: Real-time inference using camera-captured images, 1: Offline inference using a pre-loaded image |
| feed_image_path       | ./config/image.png              | The input image, which can be in formats such as png, jpg, etc.                                    |
| is_sync_mode          | 0                               | 0: Asynchronous inference, 1: Synchronous inference                                                |
| model_file_name       | ./config/faceLandmark106pts.hbm | The model file                                                                                     |
| is_shared_mem_sub     | 1                               | 0: Do not use shared memory communication, 1: Use shared memory communication                      |
| dump_render_img       | 0                               | 0: Do not save the rendered image, 1: Save the rendered image                                      |
| ai_msg_pub_topic_name | /face_landmarks_detection       | The topic name for publishing ai_msgs, only published during real-time inference                   |
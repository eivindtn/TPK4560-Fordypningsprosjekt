# Zivid and Projector Pair Calibration
## Table of contents
* [General info](#general-info)
* [Setup](#setup)
* [Procedure](#procedure)
## General Info
This repository presents a method for pair calibration of a depth camera(Zivid) with an external projector without using a printed checkerboard. The output of the calibration is the projector intrinsics and the extrinsics parameters. The code is inspired from this article [Bingyao Huang](https://bingyaohuang.github.io/Calibrate-Kinect-and-projector/).
## Setup
### Zivid SDK
To use the Zivid camera you need to download and install the "Zivid core" package. Zivid SDK Version used here was 1.8.1 but 2.1 will also work.

Follow the guide here [zivid-python](https://github.com/zivid/zivid-python) to install the SDK and zivid-pyhton.
### Package Dependicies
To install the package dependicies
```
pip install setup.py
```
## Procedure
The goal of this calibration was to have an setup that not require a printed checkerboard to calibrate both the intrinsics parameters of the projector and the extrinsic parameters between the projector and the Zivid depth camera.
### Projected checkerboard image
In order to make a checkerboard image you need to change the size of the image to your external projector resoultion with a desired colums, rows and square size.
Run this script to generate a .png file of a pattern:
```
python gen_pattern.py
```
The checkerboard pattern generated are based on this [tutorial](https://docs.opencv.org/master/da/d0d/tutorial_camera_calibration_pattern.html) and inspired code used are [code](https://github.com/opencv/opencv/blob/master/doc/pattern_tools/gen_pattern.py). Since `<gen_pattern.py>` use `<svgfig.py>` to generate a vector file(.svg) it is needed to convert the .svg file to .png format.
### Projector Calibration
To start the calibration we need to show the generated checkerboard image in the projector as in the gif below. Then we project this checkerboard either onto a plane or a wall. 

<img src="https://github.com/eivindtn/TPK4560-Specalization-Project/blob/main/images/gif_setup.gif" width="1000">

#### Capture Point Cloud and Images
The zivid allows us to capture a file (.zdf) containing both formats: point cloud(.ply) and the (.png). Since the calibration is based on [Zhang's method](https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html#calibratecamera) we need to have at least 3 different poses. My recomedation is to use at least 20 poses to obtain a more accurate result.

I did this using the zivid studio and saved the .zdf files and images .png in the folders [zdf](https://github.com/eivindtn/TPK4560-Specalization-Project/tree/main/projector-calibration/zdf) and [captured_images](https://github.com/eivindtn/TPK4560-Specalization-Project/tree/main/projector-calibration/captured_images). Another way ïs to capture and convert the .zdf file using zivids functions for this. 

#### Processing
The explaination of code and theory behind is desscribed in the [Jupyter noterbook](https://github.com/eivindtn/TPK4560-Specalization-Project/blob/main/projector-calibration/Zivid_Paired_Projector_Calibration.ipynb).
# Zivid and Projector Pair Calibration
## Table of contents
* [General info](#general-info)
* [Setup](#setup)
## General info
This repository presents a method for pair calibration of a depth camera(Zivid) with an external projector without using a printed checkerboard. The output of the calibration is the projector intrinsics and the extrinsics parameters. The code is inspired from this article [Bingyao Huang](https://bingyaohuang.github.io/Calibrate-Kinect-and-projector/).
## Setup
### Zivid SDK
To use the Zivid camera you need to download and install the "Zivid core" package. Zivid SDK Version used here was 1.8.1 but 2.1 will also work.

Follow the guide here [zivid-python](https://github.com/zivid/zivid-python) to install the SDK and zivid-pyhton.
### Package dependicies
To install the packages to runt the projector calibration script you need to install. 


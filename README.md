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
### Package dependicies
To install the package dependicies
```
pip install setup.py
```
## Procedure
The goal of this calibration was to have easy setup that can 
### Projected checkerboard image
In order to make a checkerboard image you need to change the size of the image to your external projector resoultion with a desired colums, rows and square size.
Run this script to generate a .png file of a pattern:
```
python gen_pattern.py
```
### Projector Calibration
To start the calibration we need to show the generated checkerboard image in the projector as in the image below. Then we project this checkerboard either onto a plane or a wall. 

<img src="https://github.com/eivindtn/TPK4560-Specalization-Project/blob/main/images/Test%20Jig/2.jpg" width="1000">
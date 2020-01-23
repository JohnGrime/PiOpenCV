# PiOpenCV

Scripts to install and build [`OpenCV`](https://opencv.org/) on a [Raspberry Pi](https://www.raspberrypi.org/), and an example image recognition program to test it.

Scripts based on e.g. https://docs.opencv.org/4.2.0/d7/d9f/tutorial_linux_install.html

## Fetch / build / install OpenCV

To install and build OpenCV, run the `build_opencv_linux.sh` script:

```
./build_opencv_linux.sh
```

This script will attempt to download the OpenCV sources (and the `opencv_contrib` extra modules), build them, and install the resultant OpenCV libraries on the local system.

## Example OpenCV test code

This repository includes a simple test program to check if the OpenCV installation was successful. The example program reads an input image, and attempts to find that image in a different image or live webcam video.

To build and test the example code:

```
g++ \
-I/usr/local/include/opencv4 \
-lopencv_core -lopencv_highgui -lopencv_imgproc \
-lopencv_imgcodecs -lopencv_videoio -lopencv_calib3d \
-lopencv_features2d -lopencv_xfeatures2d \
-std=c++11 -Wall -Wextra -pedantic -O2 \
example.cpp
```

If the program is run with no arguments, a brief user guide is printed:

```
$ ./a.out 

Usage : ./a.out find=path [in=path[:scale[:webcamIndex]]] [using=x] [min=N] [every=N] [gray=yes|no] [superpose=yes|no]

Where:

  find  : path to image to detect
  in    : OPTIONAL path to image in which to search (default: 'webcam', i.e. use webcam feed)
  using : OPTIONAL algorithm to use, one of 'SURF', 'SIFT', or 'ORB' (default: SIFT)
  min   : OPTIONAL minimum N matching features before bounding box drawn (default: 4)
  every : OPTIONAL run processing every N frames (default: 1)
  gray  : OPTIONAL use grayscale images (default: yes)
  superpose  : OPTIONAL flag to superpose reference image onto scene (default: no)

Notes:

The SURF and ORB algorithms can be accompanied with algorithm-specific data;
  - for SURF, this is the Hessian tolerance e.g. 'using=SURF:400' (default value: 400')
  - for ORB, this is the number of features e.g. 'using=ORB:500' (default value: 500')

The 'in' parameter can be decorated with a scale value for the data, e.g.: in=webcam:0.5,
in=mypic.png:1.5. The default scale value is 1.0 (i.e., no scaling will be performed).
If webcam use is specified, a further webcam index can be provided as a third parameter,
e.g. in=webcam:1.0:0 (default: 0).

```

## Notes:

1. See comment regarding Pi swapfile size: https://linuxize.com/post/how-to-install-opencv-on-raspberry-pi/
2. See configure output regarding adding "${BUILD_DIR}/python_loader" to $PYTHONPATH for development

The most time consuming stage is building the OpenCV library (~40 mins on RPi 4).

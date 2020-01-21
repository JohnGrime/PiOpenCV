# PiOpenCV

Scripts to install and build [`OpenCV`](https://opencv.org/) on a [Raspberry Pi](https://www.raspberrypi.org/).

Based on e.g. https://docs.opencv.org/4.2.0/d7/d9f/tutorial_linux_install.html

To install and build OpenCV:

```
./build_opencv_linux.sh
```

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

## Notes:

1. See comment regarding Pi swapfile size: https://linuxize.com/post/how-to-install-opencv-on-raspberry-pi/
2. See configure output regarding adding "${BUILD_DIR}/python_loader" to $PYTHONPATH for development

Most time consuming stage is building the OpenCV library (~40 mins on RPi 4).

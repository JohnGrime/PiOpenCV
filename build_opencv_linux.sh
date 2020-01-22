#!/usr/bin/env bash

#
# Based on e.g. https://docs.opencv.org/4.2.0/d7/d9f/tutorial_linux_install.html
#
# Notes:
#
# 1. See comment regarding Pi swapfile size: https://linuxize.com/post/how-to-install-opencv-on-raspberry-pi/
# 2. See configure output regarding adding "${BUILD_DIR}/python_loader" to $PYTHONPATH for development
#
# Most time consuming stage is Build (~40 mins on RPi 4)
#

CURRENT_DIR=$(pwd)
OPENCV_DIR="${CURRENT_DIR}/OpenCV"
SOURCE_DIR="${OPENCV_DIR}/opencv"
BUILD_DIR="${SOURCE_DIR}/build"

function Main
{
	echo ""
	echo "OpenCV installation & setup!"
	echo ""

	PrepareSystem
	GetSources
	Configure
	Build
	Install
	UpdateEnvironment
}

function CheckDirectory
{
	[[ ! -d "${1}" ]] && echo "Missing directory: ${1}" && exit
}

function PrepareSystem
{
	sudo apt update
	sudo apt upgrade
	sudo apt autoremove

	# Required: Ensure compiler toolchainpresent and up-to-date
	sudo apt-get install build-essential

	# Required: Basic platform prep
	sudo apt-get install \
		cmake \
		git \
		libgtk2.0-dev \
		pkg-config \
		libavcodec-dev \
		libavformat-dev \
		libswscale-dev

	# Optional: Python bindings, acceleration, image formats
	sudo apt-get install \
		python-dev \
		python-numpy \
		libtbb2 \
		libtbb-dev \
		libjpeg-dev \
		libpng-dev \
		libtiff-dev \
		libjasper-dev \
		libdc1394-22-dev
}

function GetSources
{
	mkdir -p "${OPENCV_DIR}"
	cd "${OPENCV_DIR}"

	git clone https://github.com/opencv/opencv.git
	git clone https://github.com/opencv/opencv_contrib.git
}

function Configure
{
	CheckDirectory "${SOURCE_DIR}"

	mkdir -p "${BUILD_DIR}"
	cd "${BUILD_DIR}"

	cmake \
	    -D CMAKE_BUILD_TYPE=RELEASE \
	    -D CMAKE_INSTALL_PREFIX=/usr/local \
	    -D OPENCV_EXTRA_MODULES_PATH="${OPENCV_DIR}/opencv_contrib/modules" \
	    -D OPENCV_ENABLE_NONFREE=ON \
	    -D BUILD_PERF_TESTS=OFF \
	    -D BUILD_TESTS=OFF \
	    ..
}

function Build
{
	CheckDirectory "${BUILD_DIR}"

	cd "${BUILD_DIR}"
	sudo make -j4
}

function Install
{
	CheckDirectory "${BUILD_DIR}"

	cd "${BUILD_DIR}"
	sudo make install
}

# Ensure we have the required environment variables to build OpenCV programs.
# If we don't do this, we might find errors like:
# "error while loading shared libraries: libopencv_core.so.4.2: cannot open shared object file: No such file or directory"
function UpdateEnvironment
{
	CheckDirectory "${BUILD_DIR}"

	PROFILE="${HOME}/.profile"

	echo "" >> ${PROFILE}
	echo "# John - added to ensure OpenCV libraries are found" >> ${PROFILE}
	echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/usr/local/lib" >> ${PROFILE}

	source ${PROFILE}
}

Main "$@" ; exit

# trustid_image_processing_cpp

## Getting Started
To make use of this library (to compile the examples, to only run them you only need the compiled files)
- Install [Intel Math Kernel Library](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-download.html) for faster computations;
- Download [CMake](https://cmake.org/) and a [C++ compiler for Windows](https://aka.ms/vs/17/release/vs_BuildTools.exe);
- Download OpenCV and dlib and put them in the depends folder:
    - For [OpenCV](https://opencv.org/releases/), please download the pre-compiled binaries and put them in the depends/opencv folder;
    - For [dlib](https://github.com/davisking/dlib), download the library source and put it in depends/dlib folder.
- Run the following command to compile the library and the examples:
```bash
mkdir build
cd build 
cmake ..
cmake --build .
```

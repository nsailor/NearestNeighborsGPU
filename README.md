# OpenCL Nearest Neighbors Benchmark

# Building

Prerequisites:
* OpenCL development environment and an OpenCL capable GPU (>= 1.2).
* CMake
* A C++11 compiler

Steps:
```
mkdir build
cd build
cmake ..
make
```

# Running

To run the application switch back to the source directory and run:

```
build/NearestNeighbor <problem size> <grid size>
```

The application will run the algorithm and measure its performance, while verifying the results.
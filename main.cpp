#define __CL_ENABLE_EXCEPTIONS
#include "cl.hpp"
#include "clutil.hpp"
#include "sort.hpp"
#include <fstream>
#include <iostream>
#include <random>
#include <sys/time.h>
#include <vector>

struct Point3 {
  float x, y, z;
};

std::vector<Point3> generate_dataset(size_t n) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(0.0, 1.0);
  std::vector<Point3> data(n);
  for (size_t i = 0; i < n; i++) {
    data[i].x = dis(gen);
    data[i].y = dis(gen);
    data[i].z = dis(gen);
  }
  return data;  // C++1x move semantics should prevent a deep copy here.
}

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "Usage: " << argv[0] << " <problem size> <grid size>\n";
    return 1;
  }

  int problem_size = 1 << atoi(argv[1]);
  int grid_size = 1 << atoi(argv[2]);
  std::cout << "Problem size: " << problem_size << " - Grid size: " << grid_size
            << std::endl;

  std::vector<cl::Platform> platforms;
  cl::Platform::get(&platforms);
  if (platforms.size() == 0) {
    std::cerr << "OpenCL not supported.\n";
    return 1;
  }

  std::vector<cl::Device> devices;
  platforms.front().getDevices(CL_DEVICE_TYPE_GPU, &devices);

  if (devices.size() == 0) {
    std::cout << "No OpenCL-capable GPU found.\n";
    return 1;
  }

  auto gpu = devices.front();

  std::cout << "Device: " << gpu.getInfo<CL_DEVICE_NAME>() << std::endl;

  cl::Context context(gpu);
  cl::CommandQueue queue(context);

  std::vector<Point3> dataset = generate_dataset(problem_size);

  struct timeval startwtime, endwtime;
  double seq_time;

  try {
    gettimeofday(&startwtime, NULL);
    gettimeofday(&endwtime, NULL);
  } catch (const cl::Error& error) {
    std::cerr << "Error: " << error.what() << " - "
              << getErrorString(error.err()) << std::endl;
    return 1;
  }

  seq_time = (double) ((endwtime.tv_usec - startwtime.tv_usec) / 1.0e6 +
                       endwtime.tv_sec - startwtime.tv_sec);
  std::printf("Time: %.3f s\n", seq_time);
}

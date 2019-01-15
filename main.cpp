#define __CL_ENABLE_EXCEPTIONS
#include "cl.hpp"
#include "clutil.hpp"
#include "nn_parallel.hpp"
#include "nn_serial.hpp"
#include "perf_timer.hpp"
#include <cassert>
#include <iostream>

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "Usage: " << argv[0] << " <problem size> <grid size>\n";
    return 1;
  }

  cl::Context context;
  try {
    context = cl::Context(CL_DEVICE_TYPE_GPU);
  } catch (const cl::Error& error) {
    std::clog
        << "Failed to find an OpenCL-capable GPU. Using the CPU instead.\n";
    context = cl::Context(CL_DEVICE_TYPE_CPU);
  }

  auto gpu = context.getInfo<CL_CONTEXT_DEVICES>().front();

  std::cout << "Device: " << gpu.getInfo<CL_DEVICE_NAME>() << std::endl;
  std::cout << "Global memory: "
            << readable_bytes(gpu.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>())
            << " - Max memory allocation size: "
            << readable_bytes(gpu.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>())
            << std::endl;
  std::cout << "---------------------------------------------------------\n";

  cl::CommandQueue queue(context);

  int problem_size = 1 << atoi(argv[1]);
  int grid_size = 1 << atoi(argv[2]);
  std::cout << "Problem size: " << problem_size
            << " points - Grid size: " << grid_size << std::endl;

  perf_timer serial_timer;
  perf_timer parallel_timer;
  try {
    using namespace nn;

    std::vector<Point3> dataset = generate_dataset(problem_size);
    std::vector<Point3> queries = generate_dataset(problem_size);

    auto serial_results =
        serial::nearest_neighbors(dataset, queries, grid_size, serial_timer);
    auto parallel_results = parallel::nearest_neighbors(
        context, queue, dataset, queries, grid_size, parallel_timer);

    std::cout << "Verifying parallel results..." << std::endl;
    for (size_t i = 0; i < queries.size(); i++) {
      if (parallel_results[i] != serial_results[i]) {
        std::cerr << "Result mismatch; CPU says " << serial_results[i]
                  << " while GPU says " << parallel_results[i] << std::endl;
      }
    }

  } catch (const cl::Error& error) {
    std::cerr << "Error: " << error.what() << " - "
              << getErrorString(error.err()) << std::endl;
    return 1;
  }

  std::cout << "Serial performance:\n";
  serial_timer.print_results();
  std::cout << "Parallel performance:\n";
  parallel_timer.print_results();
}

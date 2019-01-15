#define __CL_ENABLE_EXCEPTIONS
#include "cl.hpp"
#include "clutil.hpp"
#include "nn_serial.hpp"
#include "perf_timer.hpp"
#include <cassert>
#include <iostream>

class BoxFinder : KernelAlgorithm {
 public:
  BoxFinder(cl::Context& _context) : KernelAlgorithm(_context) {
    load_program("boxes.cl");
    map_to_boxes_kernel = cl::Kernel(program, "map_to_boxes");
  }

  void map_to_boxes(cl::CommandQueue& queue, std::vector<nn::Point3>& points,
                    std::vector<nn::int2_t>& box_map, int grid_size) {
    cl::Buffer point_buf(context, CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY,
                         sizeof(nn::Point3) * points.size());
    queue.enqueueWriteBuffer(point_buf, false, 0,
                             sizeof(nn::Point3) * points.size(), &points[0]);
    cl::Buffer boxes_buf(context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY,
                         sizeof(int) * 2 * box_map.size());
    map_to_boxes_kernel.setArg(0, point_buf);
    map_to_boxes_kernel.setArg(1, boxes_buf);
    map_to_boxes_kernel.setArg(2, grid_size);
    queue.enqueueNDRangeKernel(map_to_boxes_kernel, 0, points.size());
    queue.enqueueReadBuffer(boxes_buf, false, 0,
                            sizeof(int) * 2 * box_map.size(),
                            box_map[0].data());
  }

 protected:
  cl::Kernel map_to_boxes_kernel;
};

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

  perf_timer cpu_timer;
  try {
    using namespace nn;
    BoxFinder box_finder(context);

    std::vector<Point3> dataset = generate_dataset(problem_size);
    std::vector<Point3> queries = generate_dataset(problem_size);

    auto serial_results =
        serial::nearest_neighbors(dataset, queries, grid_size, cpu_timer);

    std::cout << "Verifying nearest neighbors..." << std::endl;
    for (size_t i = 0; i < queries.size(); i++) {
      verify_nearest_neighbor(queries[i], serial_results[i], dataset,
                              grid_size);
    }

  } catch (const cl::Error& error) {
    std::cerr << "Error: " << error.what() << " - "
              << getErrorString(error.err()) << std::endl;
    return 1;
  }

  cpu_timer.print_results();
}

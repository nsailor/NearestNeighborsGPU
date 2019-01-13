#define __CL_ENABLE_EXCEPTIONS
#include "cl.hpp"
#include "clutil.hpp"
#include "sort.hpp"
#include <array>
#include <cassert>
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

int box_for_point(const Point3& point, int d) {
  float grid_step = 1.0f / (float) d;
  int x_box = point.x / grid_step;
  int y_box = point.y / grid_step;
  int z_box = point.z / grid_step;
  return z_box * d * d + y_box * d + x_box;
}

void verify_box_mapping(const std::vector<Point3>& dataset,
                        const std::vector<int2_t>& map, int grid_size) {
  for (size_t i = 0; i < map.size(); i++) {
    static int last_box = -1;
    int point_index = map[i][1];
    int cpu_box = box_for_point(dataset[point_index], grid_size);
    int gpu_box = map[i][0];
    assert(cpu_box == gpu_box);
    assert(gpu_box >= last_box);
    last_box = gpu_box;
  }
}

class BoxFinder : KernelAlgorithm {
 public:
  BoxFinder(cl::Context& _context) : KernelAlgorithm(_context) {
    load_program("boxes.cl");
    map_to_boxes_kernel = cl::Kernel(program, "map_to_boxes");
  }

  void map_to_boxes(cl::CommandQueue& queue, std::vector<Point3>& points,
                    std::vector<int2_t>& box_map, int grid_size) {
    size_t buffer_size = sizeof(Point3) * points.size();
    cl::Buffer point_buf(context, CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY,
                         sizeof(Point3) * points.size());
    queue.enqueueWriteBuffer(point_buf, false, 0,
                             sizeof(Point3) * points.size(), &points[0]);
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
  std::cout << "Global memory: "
            << readable_bytes(gpu.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>())
            << " - Max memory allocation size: "
            << readable_bytes(gpu.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>())
            << std::endl;
  std::cout << "---------------------------------------------------------\n";

  cl::Context context(gpu);
  cl::CommandQueue queue(context);

  struct timeval startwtime, endwtime;
  double seq_time;

  int problem_size = 1 << atoi(argv[1]);
  int grid_size = 1 << atoi(argv[2]);
  std::cout << "Problem size: " << problem_size
            << " points - Grid size: " << grid_size << std::endl;

  try {
    BoxFinder box_finder(context);
    Sorter sorter(context);

    std::vector<Point3> dataset = generate_dataset(problem_size);

    // Generate the box numbers.
    std::vector<int2_t> boxes(dataset.size());
    std::cout << "Processing..." << std::endl;
    std::cout.flush();
    gettimeofday(&startwtime, NULL);
    box_finder.map_to_boxes(queue, dataset, boxes, grid_size);
    queue.finish();

    // @note Apart from not being entirely correct, our bitonic sort
    // implementation seems to be way slower than the default CPU-only sorting
    // function.
    std::sort(boxes.begin(), boxes.end(),
              [](const int2_t& a, const int2_t& b) { return a[0] < b[0]; });

    gettimeofday(&endwtime, NULL);

    std::cout << "Verifying..." << std::endl;
    verify_box_mapping(dataset, boxes, grid_size);

  } catch (const cl::Error& error) {
    std::cerr << "Error: " << error.what() << " - "
              << getErrorString(error.err()) << std::endl;
    return 1;
  }

  seq_time = (double) ((endwtime.tv_usec - startwtime.tv_usec) / 1.0e6 +
                       endwtime.tv_sec - startwtime.tv_sec);
  std::printf("Processing time: %.3f s - At least %.2f MElements/s\n", seq_time,
              problem_size / seq_time * 0.000001f);
}

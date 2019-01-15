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
  float operator[](size_t i) const {
    switch (i) {
      case 0:
        return x;
      case 1:
        return y;
      case 2:
        return z;
      default:
        throw std::logic_error("Invalid index in Point3.");
    }
  }
};
static_assert(std::is_pod<Point3>(), "Point3 is not a POD!");

std::vector<Point3> generate_dataset(int n) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(0.0, 1.0);
  std::vector<Point3> data(n);
  for (int i = 0; i < n; i++) {
    data[i].x = dis(gen);
    data[i].y = dis(gen);
    data[i].z = dis(gen);
    // If any of the components is 1.0, generate a new point
    // This is to address https://bugs.llvm.org/show_bug.cgi?id=18767
    if (((data[i].x) == 1.0) || (data[i].y == 1.0) || (data[i].z == 1.0)) {
      i--;
      continue;
    }
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
  int box_count = grid_size * grid_size * grid_size;
  for (size_t i = 0; i < map.size(); i++) {
    static int last_box = -1;
    int point_index = map[i][1];
    int cpu_box = box_for_point(dataset[point_index], grid_size);
    int gpu_box = map[i][0];
    assert(cpu_box == gpu_box);
    assert(gpu_box >= last_box);
    assert(gpu_box < box_count);
    last_box = gpu_box;
  }
}

std::vector<int2_t> create_box_index(const std::vector<int2_t>& map,
                                     int grid_size) {
  std::vector<int2_t> indices(grid_size * grid_size * grid_size, {0, 0});
  int current_box = 0;
  int current_size = 0;
  for (size_t i = 0; i < map.size(); i++) {
    if (current_box == map[i][0]) {
      current_size++;
    } else {
      indices[current_box][0] = i - current_size;
      indices[current_box][1] = current_size;
      current_box = map[i][0];
      current_size = 1;
    }
  }
  // Write the last element
  if (current_size > 0) {
    indices[current_box][0] = map.size() - current_size;
    indices[current_box][1] = current_size;
  }
  return indices;
}

void verify_box_index(const std::vector<int2_t>& index,
                      const std::vector<int2_t>& map) {
  for (size_t i = 0; i < index.size(); i++) {
    int start = index[i][0];
    int size = index[i][1];
    for (int j = 0; j < size; j++) {
      assert(map[j + start][0] == (int) i);
    }

    int item_count = 0;
    for (int2_t point : map) {
      if (point[0] == (int) i)
        item_count++;
    }
    assert(item_count == size);
  }
}

inline float point_distance(const Point3& a, const Point3& b) {
  float delta_x = a.x - b.x;
  float delta_y = a.y - b.y;
  float delta_z = a.z - b.z;
  return sqrtf(delta_x * delta_x + delta_y * delta_y + delta_z * delta_z);
}

std::pair<int, float> nearest_neighbor_in_box(
    const Point3& q, const std::vector<int2_t>& map, int2_t box,
    const std::vector<Point3>& points) {
  int box_start = box[0];
  int box_size = box[1];
  int nearest_point = -1;
  float nearest_distance = 10.0f;
  for (int i = box_start; i < (box_start + box_size); i++) {
    int point_index = map[i][1];
    assert(map[i][0] == map[box[0]][0]);
    float new_distance = point_distance(points[point_index], q);
    if (new_distance < nearest_distance) {
      nearest_distance = new_distance;
      nearest_point = point_index;
    }
  }
  return std::make_pair(nearest_point, nearest_distance);
}

inline std::array<int, 3> coordinates_for_box(int box, int d) {
  int z = box / (d * d);
  box = box % (d * d);
  int y = box / d;
  int x = box % d;
  return {x, y, z};
}

inline int box_for_coordinates(int x, int y, int z, int d) {
  return x + y * d + z * d * d;
}

// Simple unit test
void test_box_coordinates() {
  for (int i = 0; i < 6; i++) {
    int d = 1 << i;
    for (int x = 0; x < d; x++) {
      for (int y = 0; y < d; y++) {
        for (int z = 0; z < d; z++) {
          int box_number = box_for_coordinates(x, y, z, d);
          auto new_coords = coordinates_for_box(box_number, d);
          assert(new_coords[0] == x);
          assert(new_coords[1] == y);
          assert(new_coords[2] == z);
        }
      }
    }
  }
}

std::array<Point3, 2> find_domain_boundary(const std::array<int, 3>& box_origin,
                                           const std::array<int, 3>& box_size,
                                           int grid_size) {
  float grid_step = 1.0f / grid_size;
  Point3 origin;
  Point3 dest;
  origin.x = box_origin[0] * grid_step;
  origin.y = box_origin[1] * grid_step;
  origin.z = box_origin[2] * grid_step;
  dest.x = origin.x + box_size[0] * grid_step;
  dest.y = origin.y + box_size[1] * grid_step;
  dest.z = origin.z + box_size[2] * grid_step;
  return {origin, dest};
}

int nearest_neighbor(const Point3& q, const std::vector<int2_t>& map,
                     const std::vector<int2_t>& index,
                     const std::vector<Point3>& points, const int grid_size) {
  int start_box = box_for_point(q, grid_size);

  std::pair<int, float> current_nn = std::make_pair(-1, 10.0f);

  std::array<int, 3> domain_origin, domain_size;
  domain_origin = coordinates_for_box(start_box, grid_size);
  domain_size = {1, 1, 1};
  while (true) {
    for (int x = 0; x < domain_size[0]; x++) {
      for (int y = 0; y < domain_size[1]; y++) {
        for (int z = 0; z < domain_size[2]; z++) {
          int abs_x = x + domain_origin[0];
          int abs_y = y + domain_origin[1];
          int abs_z = z + domain_origin[2];
          int2_t box =
              index[box_for_coordinates(abs_x, abs_y, abs_z, grid_size)];
          if (box[1] == 0)
            continue;  // Skip empty boxes
          auto new_nn = nearest_neighbor_in_box(q, map, box, points);
          if (new_nn.second < current_nn.second) {
            current_nn = new_nn;
          }
        }
      }
    }
    // Test the edges
    bool done = true;
    for (size_t i = 0; i < 3; i++) {
      float positive_end =
          (float) (domain_origin[i] + domain_size[i]) / (float) grid_size;
      float p_dist = positive_end - q[i];
      assert(p_dist >= 0.0f);
      if ((p_dist < current_nn.second) &&
          (domain_origin[i] + domain_size[i] < grid_size)) {
        domain_size[i]++;
        done = false;
      }
      float negative_end = (float) domain_origin[i] / grid_size;
      float n_dist = q[i] - negative_end;
      assert(n_dist >= 0.0f);
      if ((n_dist < current_nn.second) && (domain_origin[i] > 0)) {
        domain_size[i]++;
        domain_origin[i]--;
        done = false;
      }
    }

    if (done)
      break;
  }

  return current_nn.first;
}

void verify_nearest_neighbor(const Point3& q, const int nn,
                             const std::vector<Point3>& points,
                             const int grid_size) {
  float nn_distance = point_distance(q, points[nn]);
  std::pair<int, float> current_nn = std::make_pair(-1, __FLT_MAX__);
  for (size_t i = 0; i < points.size(); i++) {
    float distance = point_distance(q, points[i]);
    if (distance < current_nn.second)
      current_nn = std::make_pair(i, distance);
  }
  if (current_nn.first != nn) {
    std::printf(
        "Wrong neighbor, %f for %d < %f for %d - Qbox: %d - Cbox: %d - Tbox: "
        "%d\n",
        current_nn.second, current_nn.first, nn_distance, nn,
        box_for_point(q, grid_size), box_for_point(points[nn], grid_size),
        box_for_point(points[current_nn.first], grid_size));
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
    box_finder.map_to_boxes(queue, dataset, boxes, grid_size);
    queue.finish();

    // @note Apart from not being entirely correct, our bitonic sort
    // implementation seems to be way slower than the default CPU-only
    // sorting function.
    std::sort(boxes.begin(), boxes.end(),
              [](const int2_t& a, const int2_t& b) { return a[0] < b[0]; });

#if 0
    for (size_t i = 0; i < dataset.size(); i++) {
      std::printf("%zu) (%f, %f, %f) - ", i, dataset[i].x, dataset[i].y,
                  dataset[i].z);
      int box = box_for_point(dataset[i], grid_size);
      auto box_coords = coordinates_for_box(box, grid_size);
      std::printf(" box: %d - bx: %d, by: %d, bz: %d\n", box, box_coords[0],
                  box_coords[1], box_coords[2]);
    }
#endif

    std::vector<int2_t> indices = create_box_index(boxes, grid_size);
#if 0
    for (size_t i = 0; i < indices.size(); i++) {
      std::printf("Box %zu starts at %d and has size %d.\n", i, indices[i][0],
                  indices[i][1]);
    }
#endif

    std::cout << "Verifying construction..." << std::endl;
    verify_box_mapping(dataset, boxes, grid_size);
    verify_box_index(indices, boxes);

    std::vector<Point3> queries = generate_dataset(problem_size);
    std::vector<int> neighbors(problem_size);

    std::cout << "Finding nearest neighbors..." << std::endl;
    gettimeofday(&startwtime, NULL);
    std::transform(queries.begin(), queries.end(), neighbors.begin(),
                   [&](const Point3& q) {
                     return nearest_neighbor(q, boxes, indices, dataset,
                                             grid_size);
                   });
    gettimeofday(&endwtime, NULL);
    std::cout << "Verifying nearest neighbors..." << std::endl;
    for (size_t i = 0; i < queries.size(); i++) {
      verify_nearest_neighbor(queries[i], neighbors[i], dataset, grid_size);
    }

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

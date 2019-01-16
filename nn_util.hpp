#pragma once
#include <array>
#include <cassert>
#include <exception>
#include <random>
#include <vector>

namespace nn {

typedef std::array<int, 2> int2_t;

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

inline float point_distance(const Point3& a, const Point3& b) {
  float delta_x = a.x - b.x;
  float delta_y = a.y - b.y;
  float delta_z = a.z - b.z;
  return sqrtf(delta_x * delta_x + delta_y * delta_y + delta_z * delta_z);
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

void verify_point_bins(
    const std::vector<Point3>& points,
    const std::pair<std::vector<int>, std::vector<int2_t>>& bins, int d) {
  const std::vector<int>& map = bins.first;
  const std::vector<int2_t>& index = bins.second;
  for (int i = 0; i < (int) index.size(); i++) {
    int2_t bin = index[i];
    int start = bin[0];
    int size = bin[1];
    for (int j = 0; j < size; j++) {
      int point_box = box_for_point(points[map[start + j]], d);
      assert(point_box == i);
    }

    int item_count = 0;
    for (const Point3& point : points) {
      if (box_for_point(point, d) == i)
        item_count++;
    }
    assert(item_count == size);
  }
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

}  // namespace nn
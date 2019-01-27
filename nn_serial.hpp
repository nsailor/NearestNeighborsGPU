#pragma once
#include "nn_util.hpp"
#include "perf_timer.hpp"
#include <algorithm>

namespace nn {
namespace serial {

std::pair<std::vector<int>, std::vector<int2_t>> create_point_bins(
    const std::vector<Point3>& points, int d) {
  std::vector<int> box_map(points.size());

  int box_count = d * d * d;
  int expected_density = points.size() / box_count;
  std::vector<std::vector<int>> bins(box_count);
  for (size_t i = 0; i < bins.size(); i++) {
    bins[i].reserve(expected_density * 2);
  }

  for (int i = 0; i < (int) points.size(); i++) {
    const Point3& point = points.at(i);
    int box = box_for_point(point, d);
    bins[box].push_back(i);
  }

  std::vector<int2_t> box_index(box_count);
  int current_pos = 0;
  for (int i = 0; i < box_count; i++) {
    const std::vector<int>& bin = bins[i];
    for (int j = 0; j < (int) bin.size(); j++) {
      box_map[current_pos + j] = bin[j];
    }
    int2_t box = {current_pos, (int) bin.size()};
    current_pos += (int) bin.size();
    box_index[i] = box;
  }

  return std::make_pair(box_map, box_index);
}

std::pair<int, float> nearest_neighbor_in_box(
    const Point3& q, const std::vector<int>& map, int2_t box,
    const std::vector<Point3>& points) {
  int box_start = box[0];
  int box_size = box[1];
  int nearest_point = -1;
  float nearest_distance = 10.0f;
  for (int i = box_start; i < (box_start + box_size); i++) {
    int point_index = map[i];
    float new_distance = point_distance(points[point_index], q);
    if (new_distance < nearest_distance) {
      nearest_distance = new_distance;
      nearest_point = point_index;
    }
  }
  return std::make_pair(nearest_point, nearest_distance);
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

int nearest_neighbor(const Point3& q, const std::vector<int>& map,
                     const std::vector<int2_t>& index,
                     const std::vector<Point3>& points, const int grid_size) {
  int start_box = box_for_point(q, grid_size);

  std::pair<int, float> current_nn = std::make_pair(-1, 10.0f);

  std::array<int, 3> domain_origin, domain_size;
  domain_origin = coordinates_for_box(start_box, grid_size);
  domain_size = {1, 1, 1};
  //! Exclusion zone, boxes we've already searched
  std::array<int, 3> excl_origin = domain_origin, excl_size = {0, 0, 0};
  while (true) {
    for (int x = 0; x < domain_size[0]; x++) {
      for (int y = 0; y < domain_size[1]; y++) {
        for (int z = 0; z < domain_size[2]; z++) {
          int abs_x = x + domain_origin[0];
          int abs_y = y + domain_origin[1];
          int abs_z = z + domain_origin[2];
          if ((abs_x > excl_origin[0]) &&
              (abs_x < (excl_origin[0] + excl_size[0])) &&
              (abs_y > excl_origin[1]) &&
              (abs_y < (excl_origin[1] + excl_size[1])) &&
              (abs_z > excl_origin[2]) &&
              (abs_z < (excl_origin[2] + excl_size[2])))
            continue;
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
    excl_origin = domain_origin;
    excl_size = domain_size;
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

std::vector<int> nearest_neighbors(const std::vector<Point3>& points,
                                   const std::vector<Point3>& queries,
                                   int grid_size, perf_timer& timer) {
  std::vector<int> results(points.size());

  timer.start("Creating bins");
  auto mapping = create_point_bins(points, grid_size);
  timer.end();
  // Uncomment if the algorithm start acting funny
  // verify_point_bins(points, mapping, grid_size);

  std::vector<int>& map = mapping.first;
  std::vector<int2_t>& bin_index = mapping.second;

  timer.start("Finding nearest neighbors");
  std::transform(
      queries.begin(), queries.end(), results.begin(), [&](const Point3& q) {
        return nearest_neighbor(q, map, bin_index, points, grid_size);
      });
  timer.end();

  return results;
}

}  // namespace serial
}  // namespace nn
#pragma once
#include "nn_util.hpp"
#include "perf_timer.hpp"

namespace nn {
namespace serial {

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

  // Generate the box numbers.
  std::vector<int2_t> boxes(points.size());
  timer.start("Mapping to boxes");
  for (size_t i = 0; i < points.size(); i++) {
    int box = box_for_point(points[i], grid_size);
    boxes[i] = {box, (int) i};
  }
  timer.end();

  timer.start("Sorting box map");
  std::sort(boxes.begin(), boxes.end(),
            [](const int2_t& a, const int2_t& b) { return a[0] < b[0]; });
  timer.end();

  timer.start("Creating box index");
  std::vector<int2_t> indices = create_box_index(boxes, grid_size);
  timer.end();

  timer.start("Finding nearest neighbors");
  std::transform(
      queries.begin(), queries.end(), results.begin(), [&](const Point3& q) {
        return nearest_neighbor(q, boxes, indices, points, grid_size);
      });
  timer.end();

  return results;
}

}  // namespace serial
}  // namespace nn
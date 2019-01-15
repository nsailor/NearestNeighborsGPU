#pragma once
#include "nn_serial.hpp"
#include "nn_util.hpp"
#include "perf_timer.hpp"

namespace nn {
namespace parallel {

std::vector<int> nearest_neighbors(const std::vector<Point3>& points,
                                   const std::vector<Point3>& queries,
                                   int grid_size, perf_timer& timer) {
  std::vector<int> results(points.size());

  // Generate the box numbers.
  std::vector<int2_t> boxes(points.size());
  timer.start("Mapping to boxes (serial)");
  for (size_t i = 0; i < points.size(); i++) {
    int box = box_for_point(points[i], grid_size);
    boxes[i] = {box, (int) i};
  }
  timer.end();

  timer.start("Sorting box map (serial)");
  std::sort(boxes.begin(), boxes.end(),
            [](const int2_t& a, const int2_t& b) { return a[0] < b[0]; });
  timer.end();

  timer.start("Creating box index (serial)");
  std::vector<int2_t> indices = serial::create_box_index(boxes, grid_size);
  timer.end();

  timer.start("Finding nearest neighbors (serial)");
  std::transform(
      queries.begin(), queries.end(), results.begin(), [&](const Point3& q) {
        return serial::nearest_neighbor(q, boxes, indices, points, grid_size);
      });
  timer.end();

  return results;
}

}  // namespace parallel
}  // namespace nn
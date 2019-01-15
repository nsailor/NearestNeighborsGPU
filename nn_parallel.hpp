#pragma once
#include "nn_serial.hpp"
#include "nn_util.hpp"
#include "perf_timer.hpp"

namespace nn {
namespace parallel {

std::vector<int> nearest_neighbors(cl::Context& context,
                                   cl::CommandQueue& queue,
                                   std::vector<Point3>& points,
                                   std::vector<Point3>& queries, int grid_size,
                                   perf_timer& timer) {
  std::vector<int> results(points.size());
  timer.start("Compiling the kernels");
  cl::Program program = load_program(context, "boxes.cl");
  auto map_to_boxes =
      cl::make_kernel<cl::Buffer, cl::Buffer, int>(program, "map_to_boxes");
  auto find_nearest_neighbors =
      cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer,
                      cl::Buffer, int>(program, "nearest_neighbors");
  timer.end();

  // Generate the box numbers.
  std::vector<int2_t> box_map(points.size());
  timer.start("Mapping to boxes (GPU)");
  cl::Buffer d_points(context, points.begin(), points.end(), true);
  cl::Buffer d_box_map(context, CL_MEM_READ_WRITE,
                       sizeof(int2_t) * box_map.size());
  map_to_boxes(cl::EnqueueArgs(queue, points.size()), d_points, d_box_map,
               grid_size);
  cl::copy(queue, d_box_map, box_map.begin(), box_map.end());
  queue.finish();
  timer.end();

  timer.start("Sorting box map (serial)");
  std::sort(box_map.begin(), box_map.end(),
            [](const int2_t& a, const int2_t& b) { return a[0] < b[0]; });
  timer.end();

  timer.start("Creating box index (serial)");
  std::vector<int2_t> indices = serial::create_box_index(box_map, grid_size);
  timer.end();

  timer.start("Finding nearest neighbors (GPU)");
  cl::Buffer d_queries(context, queries.begin(), queries.end(), true);
  cl::Buffer d_indices(context, indices.begin(), indices.end(), true);
  cl::Buffer d_results(context, CL_MEM_WRITE_ONLY,
                       sizeof(int) * results.size());
  // Copy the updated box map
  cl::copy(queue, box_map.begin(), box_map.end(), d_box_map);
  find_nearest_neighbors(cl::EnqueueArgs(queue, queries.size()), d_queries,
                         d_points, d_box_map, d_indices, d_results, grid_size);
  cl::copy(queue, d_results, results.begin(), results.end());
  queue.finish();
  timer.end();

  return results;
}

}  // namespace parallel
}  // namespace nn
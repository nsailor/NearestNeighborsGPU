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
  auto find_nearest_neighbors =
      cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer,
                      cl::Buffer, int>(program, "nearest_neighbors");
  timer.end();

  timer.start("Creating bins");
  auto mapping = serial::create_point_bins(points, grid_size);
  timer.end();
  std::vector<int>& map = mapping.first;
  std::vector<int2_t>& bin_index = mapping.second;

  timer.start("Finding nearest neighbors (GPU)");
  cl::Buffer d_points(context, points.begin(), points.end(), true);
  cl::Buffer d_queries(context, queries.begin(), queries.end(), true);
  cl::Buffer d_map(context, map.begin(), map.end(), true);
  cl::Buffer d_indices(context, bin_index.begin(), bin_index.end(), true);
  cl::Buffer d_results(context, CL_MEM_WRITE_ONLY,
                       sizeof(int) * results.size());
  find_nearest_neighbors(cl::EnqueueArgs(queue, queries.size()), d_queries,
                         d_points, d_map, d_indices, d_results, grid_size);
  cl::copy(queue, d_results, results.begin(), results.end());
  queue.finish();
  timer.end();

  return results;
}

}  // namespace parallel
}  // namespace nn
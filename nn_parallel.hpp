#pragma once
#include "nn_serial.hpp"
#include "nn_util.hpp"
#include "perf_timer.hpp"

namespace nn {
namespace parallel {

typedef std::pair<std::vector<int>, int> uniform_bins_t;

//! @return The point map and the query bin size
uniform_bins_t create_uniform_bins(const std::vector<Point3>& points,
                                   std::vector<int>& overflow, int d,
                                   bool no_overflow = false,
                                   int safety_factor = 2) {
  int box_count = d * d * d;
  int expected_density = points.size() / box_count;
  int bin_size =
      no_overflow ? expected_density * safety_factor : expected_density * 2;
  if (bin_size < 4)
    bin_size = 4;
  std::vector<int> map(box_count * bin_size, -1);
  std::vector<int> bin_sizes(box_count,
                             0);  //!< Temp vector to hold the number
                                  //!< of points in each bin so far.

  for (int i = 0; i < (int) points.size(); i++) {
    const Point3& point = points.at(i);
    int box = box_for_point(point, d);
    int bin_start = box * bin_size;
    if (bin_sizes[box] < bin_size) {
      map[bin_start + bin_sizes[box]] = i;
      bin_sizes[box]++;
    } else {
      if (no_overflow)
        throw std::runtime_error("Point overflow.");
      overflow.push_back(i);
    }
  }

  return std::make_pair(map, bin_size);
}

void verify_uniform_bin(const std::vector<Point3>& points,
                        const std::vector<int>& map,
                        const std::vector<int>& overflow, int bin_size, int d) {
  int box_count = d * d * d;
  for (int i = 0; i < box_count; i++) {
    int mapped_size = 0;
    int bin_start = i * bin_size;
    for (int j = 0; j < bin_size; j++) {
      if (map[bin_start + j] == -1)
        continue;
      int point_index = map[bin_start + j];
      int point_box = box_for_point(points[point_index], d);
      assert(point_box == i);
      mapped_size++;
    }
    for (int point_id : overflow) {
      if (box_for_point(points[point_id], d) == i)
        mapped_size++;
    }

    int actual_size = 0;
    for (const Point3& point : points) {
      int point_box = box_for_point(point, d);
      if (point_box == i)
        actual_size++;
    }
    assert(actual_size == mapped_size);
  }
}

std::vector<int> nearest_neighbors(cl::Context& context,
                                   cl::CommandQueue& queue,
                                   std::vector<Point3>& points,
                                   std::vector<Point3>& queries, int grid_size,
                                   perf_timer& timer) {
  std::vector<int> results(points.size());
  // Determine whether the problem can possibly fit in the local memory
  int expected_box_density =
      points.size() / (grid_size * grid_size * grid_size);
  auto device = context.getInfo<CL_CONTEXT_DEVICES>().front();
  int max_workgroup_size = device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
  std::cout << "Expected box density: " << expected_box_density
            << " - Max. work group size: " << max_workgroup_size << std::endl;
  if (expected_box_density > max_workgroup_size / 2) {
    std::clog
        << "Can't use local memory. Using slower global memory lookups instead."
        << std::endl;
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
  } else {
    // Bin the dataset
    std::vector<int> p_overflow;
    timer.start("Creating point bins");

    uniform_bins_t point_bins;
    try {
      point_bins = create_uniform_bins(points, p_overflow, grid_size, true);
    } catch (std::runtime_error& err) {
      std::cout << "Uniform binning failed - doubling the safety factor"
                << std::endl;
      // The dataset is irregular, try again with twice the margin.
      point_bins = create_uniform_bins(points, p_overflow, grid_size, true, 4);
      std::cout << "Correction succesful - expect degraded performance"
                << std::endl;
      // If we fail again, don't catch that exception, the dataset is
      // exceptionally non-uniform.
    }

    std::vector<int>& point_map = point_bins.first;
    int point_bin_size = point_bins.second;
    timer.end();

    std::vector<int> q_overflow;
    q_overflow.reserve(points.size() / 500);
    timer.start("Creating query bins");
    auto query_bins =
        create_uniform_bins(queries, q_overflow, grid_size, false);
    std::vector<int>& query_map = query_bins.first;
    q_overflow.shrink_to_fit();
    int overflow_queries = q_overflow.size();

    timer.end();
    // Uncomment if the algorithm is not behaving itself
    // verify_uniform_bin(queries, query_bins.first, overflow,
    // query_bins.second,
    //                   grid_size);

    std::cout << "Point bin size: " << point_bin_size
              << " - Query bin size: " << query_bins.second
              << " - Query Overflow: " << overflow_queries << std::endl;

    timer.start("Compiling the kernels");
    cl::Program program = load_program(context, "boxes-local.cl");
    auto find_nearest_neighbors =
        cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer,
                        cl::Buffer, int, int, cl::LocalSpaceArg,
                        cl::LocalSpaceArg>(program, "nearest_neighbors");
    timer.end();

    timer.start("Finding nearest neighbors (GPU) + Overflows (CPU)");
    cl::Buffer d_points(context, points.begin(), points.end(), true);
    cl::Buffer d_queries(context, queries.begin(), queries.end(), true);
    cl::Buffer d_map(context, point_map.begin(), point_map.end(), true);
    cl::Buffer d_query_map(context, query_map.begin(), query_map.end(), true);
    cl::Buffer d_results(context, CL_MEM_WRITE_ONLY,
                         sizeof(int) * results.size());
    std::cout << "Query map size: " << query_map.size() << std::endl;
    find_nearest_neighbors(
        cl::EnqueueArgs(queue, query_map.size(), query_bins.second), d_queries,
        d_points, d_map, d_query_map, d_results, grid_size, point_bin_size,
        cl::Local(point_bin_size * sizeof(Point3)),
        cl::Local(point_bin_size * sizeof(cl_int)));

    std::vector<int> overflow_results;
    if (overflow_queries) {
      // Process the overflows while the kernel is running.
      overflow_results.resize(overflow_queries);
      std::vector<int2_t> box_map(grid_size * grid_size * grid_size);
      for (size_t i = 0; i < box_map.size(); i++) {
        box_map[i] = {(int) i * point_bin_size, point_bin_size};
      }

      std::transform(q_overflow.begin(), q_overflow.end(),
                     overflow_results.begin(), [&](const int query) {
                       const Point3& q = queries[query];
                       return serial::nearest_neighbor(q, point_map, box_map,
                                                       points, grid_size);
                     });
    }

    cl::copy(queue, d_results, results.begin(), results.end());
    queue.finish();

    for (int i = 0; i < overflow_queries; i++) {
      int query = q_overflow.at(i);
      results[query] = overflow_results.at(i);
    }
    timer.end();
  }

  return results;
}

}  // namespace parallel
}  // namespace nn
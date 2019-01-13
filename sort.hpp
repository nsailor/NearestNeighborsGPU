#pragma once
#include "cl.hpp"
#include "clutil.hpp"

class Sorter : public KernelAlgorithm {
 public:
  Sorter(cl::Context& _context) : KernelAlgorithm(_context) {
    load_program("sort2.cl");
    kernel_local = cl::Kernel(program, "sort2_local");
    kernel_global = cl::Kernel(program, "sort2_global");
  }

  template <typename T = int>
  void sort(cl::CommandQueue& queue, std::vector<T>& array) {
    cl::Buffer buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                      array.size() * sizeof(T), array.data());
    cl_uint length = array.size() / 2;
    // On a GPU, we can use the maximum work group size.
    // A CPU however has a maximum size of 1.
    uint limit = device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
    if (device.getInfo<CL_DEVICE_TYPE>() == CL_DEVICE_TYPE_CPU)
      limit = 1;
    uint threshold = length <= limit ? length : limit;
    if (threshold > 1) {
      kernel_local.setArg(0, buffer);
      kernel_local.setArg(1, cl::Local(threshold * 2));
      auto err = queue.enqueueNDRangeKernel(kernel_local, 0, length, threshold);
    }

    kernel_global.setArg(0, buffer);
    for (cl_uint k = threshold * 2; k <= length; k *= 2) {
      for (cl_uint j = k / 2; j > 0; j /= 2) {
        kernel_global.setArg(1, k);
        kernel_global.setArg(2, j);
        queue.enqueueNDRangeKernel(kernel_global, 0, length);
      }
    }
    queue.enqueueReadBuffer(buffer, CL_TRUE, 0, array.size() * sizeof(T),
                            array.data());
  }

 protected:
  cl::Kernel kernel_local;
  cl::Kernel kernel_global;
};
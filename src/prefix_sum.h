#pragma once
void gpu_workefficient_prefixsum(cl::Context& context, cl::CommandQueue& queue, cl::Kernel& kernel, const std::vector<int>& input, std::vector<int>& output, const opencl_manager& manager);
void gpu_prefixsum(cl::Context& context, cl::CommandQueue& queue, cl::Kernel& kernel, const std::vector<int>& input, std::vector<int>& output);
void gpu_prefixsum2(cl::Context& context, cl::CommandQueue& queue, cl::Kernel& kernel, const std::vector<int>& input, std::vector<int>& output);
std::vector<int> sequential_scan_inclusive(std::vector<int> input);
std::vector<int> sequential_scan_exclusive(std::vector<int> input);
void sequential_fill_vector(const int size, std::vector<int>& v);

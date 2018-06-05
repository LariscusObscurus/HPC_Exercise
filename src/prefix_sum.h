#pragma once
void gpu_workefficient_prefixsum(cl::Context& context, cl::CommandQueue& queue, cl::Kernel& kernel, std::vector<int>& input, std::vector<int>& output, const opencl_manager& manager);
void gpu_prefixsum(cl::Context& context, cl::CommandQueue& queue, cl::Kernel& kernel, std::vector<int>& input, std::vector<int>& output);
std::vector<int> sequential_scan(std::vector<int> input);
void sequential_fill_vector(const int size, std::vector<int>& v);

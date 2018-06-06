#pragma once
#include <vector>
#include <CL/cl.hpp>

void stream_compact(cl::Context& context, cl::CommandQueue& queue, cl::Kernel& kernel, const std::vector<int>& input, std::vector<int>& output);
bool isPrime(int n);
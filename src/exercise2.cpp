#include "CL/cl.hpp"
#include <opencl_manager.h>
#include <string>
#include <functional>
#include <iostream>
#include <cassert>
#include <chrono>
#include <prefix_sum.h>

template<typename T>
void print_vector(std::vector<T> vector)
{
    std::cout << "[ ";
    for (auto& it : vector)
    {
        std::cout << it << " ";
    }
    std::cout << "]" << std::endl;
}

std::chrono::nanoseconds measure_runtime(){
	const auto start_time = std::chrono::steady_clock::now();

	const auto end_time = std::chrono::steady_clock::now();
	const auto diff = end_time - start_time;
	return diff;
}


int main(int argc, char* argv[])
{
    try
    {

        auto open_cl = opencl_manager{};
        open_cl.compile_program("scan.cl");
        open_cl.load_kernel("single_workgroup_prefixsum");
        open_cl.load_kernel("blelloch_scan");
        open_cl.load_kernel("add_groups");
        open_cl.load_kernel("naive_parallel_prefixsum");

        //Fill test vector

        auto threads = open_cl.get_max_workgroup_size();
        auto items = threads;

        auto test = std::vector<int>{};
        sequential_fill_vector(items, test);
        auto result = sequential_scan(test);

        std::function<void(cl::Context& context, cl::CommandQueue& queue, cl::Kernel& kernel, std::vector<int>&, std::vector<int>&)> fun = gpu_prefixsum;
        std::function<void(cl::Context& context, cl::CommandQueue& queue, cl::Kernel& kernel, std::vector<int>&, std::vector<int>&, opencl_manager&)> workefficient_scan = gpu_workefficient_prefixsum;

        auto output = std::vector<int>(test.size());
        //open_cl.execute_kernel("single_workgroup_prefixsum", fun, test, output);
        open_cl.execute_kernel("blelloch_scan", workefficient_scan, test, output, open_cl);
        //open_cl.execute_kernel("naive_parallel_prefixsum", fun, test, output);
        //output.insert(output.begin(), 0);//only for naive because it is a non inclusive scan

        for (auto i = 0; i < test.size(); ++i)
        {
            if (result[i] != output[i])
            {
                std::cout << "At pos " << i << " Result was: " << output[i] << " Should be: " << result[i] << std::endl;

                std::getchar();
                return -1;
            }
        }
        std::cout << "GPU Result OK. " << std::endl;
    }
    catch (std::runtime_error ex)
    {
        std::cout << ex.what() << std::endl;
    }

    std::getchar();
    return 0;
}

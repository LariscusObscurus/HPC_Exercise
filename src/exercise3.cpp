
#include "opencl_manager.h"
#include <iostream>
#include "stream_compact.h"
#include "prefix_sum.h"

int main(int argc, char* argv[])
{
    try
    {

        auto open_cl = opencl_manager{};
        open_cl.compile_program("stream_compact.cl");
        open_cl.load_kernel("compact");

        //Fill test vector
        auto threads = open_cl.get_max_workgroup_size();
        auto items = threads;

        auto test = std::vector<int>();
        sequential_fill_vector(items, test);

        std::function<void(cl::Context& context, cl::CommandQueue& queue, cl::Kernel& kernel, const std::vector<int>&, std::vector<int>&)> func = stream_compact;

        auto output = std::vector<int>(test.size());
        open_cl.execute_kernel<const std::vector<int>&, std::vector<int>&>("compact", func, test, output); 

    }
    catch (std::runtime_error ex)
    {
        std::cout << ex.what() << std::endl;
    }

    std::getchar();
    return 0;
}

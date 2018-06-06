#include "opencl_manager.h"
#include <iostream>
#include "stream_compact.h"
#include "prefix_sum.h"

template<typename T>
void print_vector(std::vector<T> vector)
{
	std::cout << "[(size: " << vector.size() << ") ";
	for (auto& it : vector)
	{
		std::cout << it << " ";
	}
	std::cout << "]" << std::endl;
}

void sequential_compact( const std::vector<int>& input, std::vector<int>& output) 
{
    output.resize(input.size());

	unsigned int j = 0;
	for (auto it : input)
	{
		//it & (it-1)==0 // power of two
		//isPrime(it) // is prime number
		if (it > 10) {
			output[j] = input[it];
			j++;
		}
	}
}

int main(int argc, char* argv[])
{
    try
    {

        auto open_cl = opencl_manager{};
        open_cl.compile_program("stream_compact.cl");
		open_cl.load_kernel("compact");
		open_cl.load_kernel("compactArbitrary");

        //Fill test vector
        auto threads = 512;
        auto items = threads;

        auto test = std::vector<int>();
        sequential_fill_vector(items, test);

        auto expected_result = std::vector<int>(test.size());
        sequential_compact(test, expected_result);


        std::function<void(cl::Context& context, cl::CommandQueue& queue, cl::Kernel& kernel, const std::vector<int>&, std::vector<int>&)> func = stream_compact;
		 
        auto output = std::vector<int>(test.size());
        open_cl.execute_kernel<const std::vector<int>&, std::vector<int>&>("compactArbitrary", func, test, output); 

		print_vector(expected_result);
		print_vector(output);
    }
    catch (std::runtime_error ex)
    {
        std::cout << ex.what() << std::endl;
    }

    std::getchar();
    return 0;
}

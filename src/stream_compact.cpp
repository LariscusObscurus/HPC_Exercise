#include <opencl_manager.h>

void stream_compact(cl::Context& context, cl::CommandQueue& queue, cl::Kernel& kernel, const std::vector<int>& input, std::vector<int>& output)
{
    const auto input_buffer_size = input.size() * sizeof(int);
    const auto output_buffer_size = output.size() * sizeof(int);

    const auto input_buffer = cl::Buffer(context, CL_MEM_READ_ONLY, input_buffer_size);
    const auto output_buffer = cl::Buffer(context, CL_MEM_READ_WRITE, output_buffer_size);

    queue.enqueueWriteBuffer(input_buffer, CL_TRUE, 0, input_buffer_size, input.data());

    kernel.setArg(0, input_buffer);
    kernel.setArg(1, output_buffer);

    kernel.setArg(2, input.size());

    auto event = cl::Event{};
    auto rv = queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(input.size()), cl::NDRange(256), nullptr, &event);
    if (rv != CL_SUCCESS)
    {
        throw std::runtime_error("Could not enqueue kernel. Return value was:  " + std::to_string(rv));
    }

    queue.enqueueReadBuffer(output_buffer, CL_TRUE, 0, output_buffer_size, &output[0], nullptr, &event);

    queue.finish();
    event.wait();
}

bool isPrime(int n) {
	for (int i = 2; i <= n/2; i++) {
		if (n%i > 0)
			return false;
	}
	return true;
}

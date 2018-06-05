__kernel void blelloch_scan(__global const int* input,
    __global int* output,
    __local int * temp,
    const int items)
{
    uint global_id = get_global_id(0);
    uint local_id = get_local_id(0);

    uint group_id = get_group_id(0);
    uint group_size = get_local_size(0);

    uint depth = 1;

    if (global_id >= items) return;

    temp[local_id] = input[global_id];

    //upsweep
    for (uint stride = group_size >> 1; stride > 0; stride >>= 1) {
        barrier(CLK_LOCAL_MEM_FENCE);

        if (local_id < stride) {
            uint i = depth * (2 * local_id + 1) - 1;
            uint j = depth * (2 * local_id + 2) - 1;
            temp[j] += temp[i];
        }

        depth <<= 1;
    }

    if (local_id == 0) {
        temp[group_size - 1] = 0;
    }

    //downsweep
    for (uint stride = 1; stride < group_size; stride <<= 1) {

        depth >>= 1;
        barrier(CLK_LOCAL_MEM_FENCE);

        if (local_id < stride) {
            uint i = depth * (2 * local_id + 1) - 1;
            uint j = depth * (2 * local_id + 2) - 1;

            int t = temp[j];
            temp[j] += temp[i];
            temp[i] = t;
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    output[global_id] = temp[local_id];
}

__kernel void compact(
    __global const int* input,
    __global int* output,
    __local int* temp,
    __local int* temp_2)
{
    int local_id = get_local_id(0);
    int size = get_local_size(0);

    temp[local_id] = input[local_id] > 10;

    barrier(CLK_LOCAL_MEM_FENCE);

    blelloch_scan(output, output, temp_2, size);

    barrier(CLK_LOCAL_MEM_FENCE);

    if (temp[local_id]) {
        output[output[local_id]] = input[local_id];
    }

}

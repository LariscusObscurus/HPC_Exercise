__kernel void compact(
    __global int*input,
    __global int*output,
    __local unsigned* flag,
    __local unsigned* idx)
{
    int local_id = get_local_id(0);
    int size = get_local_size(0);

    flag[local_id] = input[local_id] > 10;

    barrier(CLK_LOCAL_MEM_FENCE);

    if (local_id < size / 2) {
        idx[2 * local_id] = flag[2 * local_id];
        idx[2 * local_id + 1] = flag[2 * local_id + 1];
    }

    int offset = 1;
    for (unsigned d = size / 2; d > 0; d /= 2) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (local_id < d) {
            int ai = offset * (2 * local_id + 1) - 1;
            int bi = offset * (2 * local_id + 2) - 1;
            idx[bi] += idx[ai];
        }
        offset *= 2;
    }
    if (local_id == 0) idx[size - 1] = 0;
    for (unsigned d = 1; d < size; d *= 2) {
        offset /= 2;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (local_id < d) {
            int ai = offset * (2 * local_id + 1) - 1;
            int bi = offset * (2 * local_id + 2) - 1;
            int temp = idx[ai];
            idx[ai] = idx[bi];
            idx[bi] += temp;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (flag[local_id]) {
        output[idx[local_id]] = input[local_id];
    };
}
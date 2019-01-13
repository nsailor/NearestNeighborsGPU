__kernel void sort2_global(__global int2 *data, uint k, uint j)
{
    unsigned int index = get_global_id(0);

    unsigned int ixj = index ^ j;
    if (ixj > index) {
        if ((index & k) == 0) {
            if (data[index].x > data[ixj].x) {
                int2 temp = data[index];
                data[index] = data[ixj];
                data[ixj] = temp;
            }
        } else {
            if (data[index].x < data[ixj].x) {
                int2 temp = data[index];
                data[index] = data[ixj];
                data[ixj] = temp;
            }
        }
    }
}

__kernel void sort2_local(__global int2 *global_data, __local int2 *data)
{
    unsigned int index = get_global_id(0);
    unsigned int length = get_local_size(0);

    data[index] = global_data[index];
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint k = 2; k <= length; k *= 2) {
        for (uint j = k / 2; j > 0; j /= 2) {
            unsigned int ixj = index ^ j;
            if (ixj > index) {
                if ((index & k) == 0) {
                    if (data[index].x > data[ixj].x) {
                        int2 temp = data[index];
                        data[index] = data[ixj];
                        data[ixj] = temp;
                    }
                } else {
                    if (data[index].x < data[ixj].x) {
                        int2 temp = data[index];
                        data[index] = data[ixj];
                        data[ixj] = temp;
                    }
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

    global_data[index] = data[index];
}
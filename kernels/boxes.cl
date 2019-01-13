/* Unfortantely, we cannot pass float3 as an argument, since OpenCL assumes that
 * it is padded as a float4. */
__kernel void map_to_boxes(__global float *points, __global int2 *map, int d)
{
    int index = get_global_id(0);
    int point_index = index * 3;
    float3 point;
    point.x = points[point_index];
    point.y = points[point_index + 1];
    point.z = points[point_index + 2];

    float grid_step = 1.0f / (float) d;
    int x_box = point.x / grid_step;
    int y_box = point.y / grid_step;
    int z_box = point.z / grid_step;
    int box = z_box * d * d + y_box * d + x_box;
    map[index].x = box;
    map[index].y = index;
}

static int box_for_point(float3 point, int d) {
    float grid_step = 1.0f / (float) d;
    int x_box = point.x / grid_step;
    int y_box = point.y / grid_step;
    int z_box = point.z / grid_step;
    return z_box * d * d + y_box * d + x_box;
}

static int3 coordinates_for_box(int box, int d) {
    int z = box / (d * d);
    box = box % (d * d);
    int y = box / d;
    int x = box % d;
    return (int3)(x, y, z);
}

static int box_for_coordinates(int3 pos, int d) {
    return pos.x + pos.y * d + pos.z * d * d;
}

typedef struct {
    int index;
    float distance;
} nn_result_t;

static nn_result_t nearest_result(nn_result_t current, nn_result_t candidate) {
    if (current.distance > candidate.distance)
        return candidate;
    return current;
}

/* Unfortantely, we cannot pass float3 as an argument, since OpenCL assumes that
 * it is padded as a float4. */
__kernel void map_to_boxes(__global float *points, __global int2 *map, int d) {
    int index = get_global_id(0);
    float3 point = vload3(index, points);

    float grid_step = 1.0f / (float) d;
    int x_box = point.x / grid_step;
    int y_box = point.y / grid_step;
    int z_box = point.z / grid_step;
    int box = z_box * d * d + y_box * d + x_box;
    map[index].x = box;
    map[index].y = index;
}

static nn_result_t nearest_neighbor_in_box(
    float3 q,
    int2 box,
    __global int2 *map,
    __global float *points) {
    nn_result_t current;
    current.index = -1;
    current.distance = 10.0f;
    int box_start = box.x;
    int box_size = box.y;
    for (int i = box_start; i < (box_start + box_size); i++) {
        nn_result_t next;
        next.index = map[i].y;
        float3 candidate = vload3(next.index, points);
        next.distance = distance(q, candidate);
        current = nearest_result(current, next);
    }
    return current;
}

__kernel void nearest_neighbors(__global float *queries,
                                __global float *points,
                                __global int2 *map,
                                __global int2 *box_index,
                                __global int *results,
                                int d) {
    int index = get_global_id(0);
    // Get the current point and find its box
    float3 q = vload3(index, queries);
    int start_box = box_for_point(q, d);

    nn_result_t current_nn;
    current_nn.index = -1;
    current_nn.distance = 10.0f;

    int3 domain_origin = coordinates_for_box(start_box, d);
    int3 domain_size = (int3)(1, 1, 1);

    while (true) {
        for (int x = 0; x < domain_size.x; x++) {
            for (int y = 0; y < domain_size.y; y++) {
                for (int z = 0; z < domain_size.z; z++) {
                    int3 abs_box = (int3)(x, y, z) + domain_origin;
                    int box_id = box_for_coordinates(abs_box, d);
                    int2 box = box_index[box_id];
                    if (box.y == 0)
                        continue; // Skip empty boxes
                    nn_result_t new_nn = nearest_neighbor_in_box(q, box, map, points);
                    current_nn = nearest_result(current_nn, new_nn);
                }
            }
        }

        // Check to see if we're done
        bool done = true;
        float3 positive_end = convert_float3(domain_origin + domain_size) / (float)d;
        float3 p_dist = positive_end - q;
        float3 negative_end = convert_float3(domain_origin) / (float)d;
        float3 n_dist = q - negative_end;
        // X
        if ((p_dist.x < current_nn.distance) &&
                (domain_origin.x + domain_size.x < d)) {
            domain_size.x++;
            done = false;
        }
        if ((n_dist.x < current_nn.distance) && (domain_origin.x > 0)) {
            domain_origin.x--;
            domain_size.x++;
            done = false;
        }

        // Y
        if ((p_dist.y < current_nn.distance) &&
                (domain_origin.y + domain_size.y < d)) {
            domain_size.y++;
            done = false;
        }
        if ((n_dist.y < current_nn.distance) && (domain_origin.y > 0)) {
            domain_origin.y--;
            domain_size.y++;
            done = false;
        }

        // Z
        if ((p_dist.z < current_nn.distance) &&
                (domain_origin.z + domain_size.z < d)) {
            domain_size.z++;
            done = false;
        }
        if ((n_dist.z < current_nn.distance) && (domain_origin.z > 0)) {
            domain_origin.z--;
            domain_size.z++;
            done = false;
        }

        if (done)
            break;
    }

    results[index] = current_nn.index;
}
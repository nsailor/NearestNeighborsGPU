
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

static nn_result_t nearest_neighbor_in_box(float3 q,
        int2 box,
        __global int *map,
        __global float *points) {
    nn_result_t current;
    current.index = -1;
    current.distance = 10.0f;
    int box_start = box.x;
    int box_size = box.y;
    for (int i = box_start; i < (box_start + box_size); i++) {
        nn_result_t next;
        next.index = map[i];
        if (next.index == -1)
            break;
        float3 candidate = vload3(next.index, points);
        next.distance = distance(q, candidate);
        current = nearest_result(current, next);
    }
    return current;
}

static nn_result_t nearest_neighbor_in_primary_candidates(float3 q,
        int point_bin_size,
        __local float *primary_candidates,
        __local int *primary_indices)
{
    nn_result_t current;
    current.index = -1;
    current.distance = 10.0f;
    for (int i = 0; i < point_bin_size; i++) {
        nn_result_t next;
        next.index = primary_indices[i];
        if (next.index == -1)
            break;
        float3 candidate = vload3(i, primary_candidates);
        next.distance = distance(q, candidate);
        current = nearest_result(current, next);
    }
    return current;
}


__kernel void nearest_neighbors(__global float *queries,
                                __global float *points,
                                __global int *map,
                                __global int *query_map,
                                __global int *results,
                                int d, int point_bin_size,
                                __local float *primary_candidates,
                                __local int *primary_indices) {
    int global_index = get_global_id(0);
    int local_size = get_local_size(0);
    int local_index = get_local_id(0);

    int local_box = global_index / local_size;

    // Here we assume that point_bin_size > local_size, which should be the
    // case given that the point bins are large enough so that they won't overflow
    // while the queries should overflow. Therefore points_per_unit >= 1.
    int points_per_unit = point_bin_size / local_size;
    int overflow_points = point_bin_size % local_size; // Extra points for the last unit.

    for (int i = 0; i < points_per_unit; i++) {
        int point_index = map[local_box * point_bin_size + local_index * points_per_unit + i];
        float3 point;
        if (point_index == -1) {
            point = (float3)(10.0f, 10.0f, 10.0f); // Out of the domain
        } else {
            point = vload3(point_index, points);
        }
        vstore3(point, local_index * points_per_unit + i, primary_candidates);
        primary_indices[local_index * points_per_unit + i] = point_index;
    }
#if 1
    // If we are the last unit, load the overflow points.
    if (local_index == local_size - 1) {
        for (int i = 0; i < overflow_points; i++) {
            int point_index = map[local_box * point_bin_size + (local_index + 1) * points_per_unit + i];
            float3 point;
            if (point_index == -1) {
                point = (float3)(10.0f, 10.0f, 10.0f);
            } else {
                point = vload3(point_index, points);
            }
            vstore3(point, (local_index + 1) * points_per_unit + i, primary_candidates);
            primary_indices[(local_index + 1) * points_per_unit + i] = point_index;
        }
    }
#endif
    barrier(CLK_LOCAL_MEM_FENCE);

    int index = query_map[global_index];
    if (index == -1) {
        return;
    }

    // Get the current point and find its box
    float3 q = vload3(index, queries);
    int start_box = box_for_point(q, d);

    nn_result_t current_nn;
    current_nn.index = -1;
    current_nn.distance = 10.0f;

    int3 domain_origin = coordinates_for_box(start_box, d);
    int3 domain_size = (int3)(1, 1, 1);
    int3 excl_origin = domain_origin;
    int3 excl_size = (int3)(0, 0, 0);
    while (true) {
        for (int x = 0; x < domain_size.x; x++) {
            for (int y = 0; y < domain_size.y; y++) {
                for (int z = 0; z < domain_size.z; z++) {
                    int3 abs_box = (int3)(x, y, z) + domain_origin;
                    if ((abs_box.x >= excl_origin.x) &&
                            (abs_box.x < excl_origin.x + excl_size.x) &&
                            (abs_box.y >= excl_origin.y) &&
                            (abs_box.y < excl_origin.y + excl_size.y) &&
                            (abs_box.z >= excl_origin.z) &&
                            (abs_box.z < excl_origin.z + excl_size.z)) {
                        continue;
                    }
                    int box_id = box_for_coordinates(abs_box, d);
                    nn_result_t new_nn;
                    if (box_id == local_box) {
                        new_nn = nearest_neighbor_in_primary_candidates(q, point_bin_size, primary_candidates, primary_indices);
                    } else {
                        int2 box = (int2)(point_bin_size * box_id, point_bin_size);
                        new_nn = nearest_neighbor_in_box(q, box, map, points);
                    }
                    current_nn = nearest_result(current_nn, new_nn);
                }
            }
        }

        excl_origin = domain_origin;
        excl_size = domain_size;

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
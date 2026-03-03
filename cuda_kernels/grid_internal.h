#pragma once
#include <cuda_runtime.h>

struct GridParams {
    int np;
    int ncell;
    int cell_num_x;
    int cell_num_y;
    int cell_num_z;
    double inv_h;
    double3 domain_min;
    double3 domain_max;
};

struct GridContext {
    GridParams params;

    int* part_cel = nullptr;
    int* part_idx = nullptr;
    int* cell_beg = nullptr;
    int* cell_end = nullptr;

    int* part_cel_tmp = nullptr;
    int* part_idx_tmp = nullptr;

    void* cub_buffer = nullptr;
    size_t cub_buffer_bytes = 0;
};

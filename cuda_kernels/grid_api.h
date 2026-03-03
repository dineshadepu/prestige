#pragma once
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void* grid_handle_t;

grid_handle_t grid_create();
void grid_destroy(grid_handle_t grid);

void grid_config(
    grid_handle_t grid,
    int cell_num_x,
    int cell_num_y,
    int cell_num_z,
    double inv_h,
    double3 domain_min,
    double3 domain_max
);

void grid_build(
    grid_handle_t grid,
    const double* d_pos,
    int np,
    int* part_idx,
    int* cell_beg,
    int* cell_end
);

#ifdef __cplusplus
}
#endif

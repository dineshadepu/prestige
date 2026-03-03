#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <cassert>
#include "grid_api.h"
#include "grid_internal.h"

#define CUDA_CHECK(x) \
do { \
    cudaError_t e = (x); \
    if (e != cudaSuccess) { \
        printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        abort(); \
    } \
} while(0)

/* ===================== KERNELS ===================== */

__global__ void compute_cell_index(
    GridParams gp,
    const double* pos,
    int* part_cel,
    int* part_idx)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= gp.np) return;

    int base = 3 * i;

    double px = pos[base + 0];
    double py = pos[base + 1];
    double pz = pos[base + 2];

    if (px < gp.domain_min.x || px >= gp.domain_max.x ||
        py < gp.domain_min.y || py >= gp.domain_max.y ||
        pz < gp.domain_min.z || pz >= gp.domain_max.z)
    {
        part_cel[i] = -1;
        part_idx[i] = i;
        return;
    }

    int cx = (int)((px - gp.domain_min.x) * gp.inv_h);
    int cy = (int)((py - gp.domain_min.y) * gp.inv_h);
    int cz = (int)((pz - gp.domain_min.z) * gp.inv_h);

    cx = max(0, min(cx, gp.cell_num_x - 1));
    cy = max(0, min(cy, gp.cell_num_y - 1));
    cz = max(0, min(cz, gp.cell_num_z - 1));

    int cell =
        cx * gp.cell_num_y * gp.cell_num_z +
        cy * gp.cell_num_z +
        cz;

    part_cel[i] = cell;
    part_idx[i] = i;
}

__global__ void find_cell_ranges(
    int np,
    int* cell_beg,
    int* cell_end,
    const int* part_cel)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= np) return;

    int cell = part_cel[i];
    if (cell < 0) return;

    if (i == 0 || part_cel[i - 1] != cell)
        cell_beg[cell] = i;

    if (i == np - 1 || part_cel[i + 1] != cell)
        cell_end[cell] = i + 1;
}

/* ===================== API ===================== */

grid_handle_t grid_create()
{
    auto* g = new GridContext();
    g->part_cel = nullptr;
    g->part_cel_tmp = nullptr;
    g->part_idx_tmp = nullptr;
    g->cub_buffer = nullptr;
    g->cub_buffer_bytes = 0;
    return g;
}

void grid_destroy(grid_handle_t handle)
{
    auto* g = (GridContext*)handle;
    if (!g) return;

    // Only free CUDA-owned temporaries
    if (g->part_cel)      cudaFree(g->part_cel);
    if (g->part_cel_tmp)  cudaFree(g->part_cel_tmp);
    if (g->part_idx_tmp)  cudaFree(g->part_idx_tmp);
    if (g->cub_buffer)    cudaFree(g->cub_buffer);

    delete g;
}

void grid_config(
    grid_handle_t handle,
    int cell_num_x,
    int cell_num_y,
    int cell_num_z,
    double inv_h,
    double3 domain_min,
    double3 domain_max)
{
    auto* g = (GridContext*)handle;

    g->params.cell_num_x = cell_num_x;
    g->params.cell_num_y = cell_num_y;
    g->params.cell_num_z = cell_num_z;
    g->params.ncell = cell_num_x * cell_num_y * cell_num_z;
    g->params.inv_h = inv_h;
    g->params.domain_min = domain_min;
    g->params.domain_max = domain_max;
}

void grid_build(
    grid_handle_t handle,
    const double* d_pos,
    int np,
    int* part_idx,
    int* cell_beg,
    int* cell_end)
{
    auto* g = (GridContext*)handle;
    g->params.np = np;

    dim3 block(128);
    dim3 grid((np + block.x - 1) / block.x);

    // Allocate or resize temporary buffers if needed
    if (!g->part_cel) {
        CUDA_CHECK(cudaMalloc(&g->part_cel, np * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&g->part_cel_tmp, np * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&g->part_idx_tmp, np * sizeof(int)));
    }

    compute_cell_index<<<grid, block>>>(
        g->params,
        d_pos,
        g->part_cel,
        part_idx);

    CUDA_CHECK(cudaGetLastError());

    // --- CUB sort ---
    size_t tmp_bytes = 0;
    cub::DeviceRadixSort::SortPairs(
        nullptr, tmp_bytes,
        g->part_cel, g->part_cel_tmp,
        part_idx, g->part_idx_tmp,
        np);

    if (tmp_bytes > g->cub_buffer_bytes) {
        if (g->cub_buffer) cudaFree(g->cub_buffer);
        CUDA_CHECK(cudaMalloc(&g->cub_buffer, tmp_bytes));
        g->cub_buffer_bytes = tmp_bytes;
    }

    cub::DeviceRadixSort::SortPairs(
        g->cub_buffer, tmp_bytes,
        g->part_cel, g->part_cel_tmp,
        part_idx, g->part_idx_tmp,
        np);

    CUDA_CHECK(cudaMemcpy(
        g->part_cel,
        g->part_cel_tmp,
        np * sizeof(int),
        cudaMemcpyDeviceToDevice));

    CUDA_CHECK(cudaMemcpy(
        part_idx,
        g->part_idx_tmp,
        np * sizeof(int),
        cudaMemcpyDeviceToDevice));

    CUDA_CHECK(cudaMemset(cell_beg, -1, g->params.ncell * sizeof(int)));
    CUDA_CHECK(cudaMemset(cell_end, -1, g->params.ncell * sizeof(int)));

    find_cell_ranges<<<grid, block>>>(
        np,
        cell_beg,
        cell_end,
        g->part_cel);

    CUDA_CHECK(cudaGetLastError());
}

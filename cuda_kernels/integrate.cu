extern "C" __global__
void dem_stage1(
    double* u,       // velocity [N*3]
    const double* force,   // force [N*3]
    const double* m,       // mass [N]
    const unsigned int* n,
    const double* half_dt
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int N = n[0];
    if (i >= N) return;

    int k = 3 * i;

    double inv_m = 1.0 / m[i];
    u[k+0] += (*half_dt) * force[k+0] * inv_m;
    u[k+1] += (*half_dt) * force[k+1] * inv_m;
    u[k+2] += (*half_dt) * force[k+2] * inv_m;
}

extern "C" __global__
void dem_stage2(
    double* x,     // position [N*3]
    const double* u,
    const unsigned int* n,
    const double* dt
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int N = n[0];
    if (i >= N) return;

    int k = 3 * i;
    x[k+0] += (*dt) * u[k+0];
    x[k+1] += (*dt) * u[k+1];
    x[k+2] += (*dt) * u[k+2];
}

extern "C" __global__
void dem_stage3(
    double* u,       // velocity [N*3]
    const double* force,   // force [N*3]
    const double* m,       // mass [N]
    const unsigned int* n,
    const double* half_dt
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int N = n[0];
    if (i >= N) return;

    int k = 3 * i;

    double inv_m = 1.0 / m[i];
    u[k+0] += (*half_dt) * force[k+0] * inv_m;
    u[k+1] += (*half_dt) * force[k+1] * inv_m;
    u[k+2] += (*half_dt) * force[k+2] * inv_m;
}
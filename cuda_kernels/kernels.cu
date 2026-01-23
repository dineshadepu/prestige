extern "C" __global__
void add_gravity(double* fy, int n, double g) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        fy[i] -= g;
    }
}

extern "C" __global__
void integrate(
    double* x, double* y,
    double* vx, double* vy,
    double* fy,
    int n, double dt
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        vy[i] += fy[i] * dt;   // update velocity
        x[i]  += vx[i] * dt;  // update position
        y[i]  += vy[i] * dt;
    }
}

#include <stdio.h>
#include <math.h>


extern "C" __global__
void reset_force(
    double* force,      // [N*3]
    const unsigned int* n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int N = n[0];
    if (i >= N) return;

    int k = 3 * i;

    force[k+0] = 0.;
    force[k+1] = 0.;
    force[k+2] = 0.;
}


extern "C" __global__
void gravity_force(
    double* force,      // [N*3]
    const double* m,    // [N]
    const double* g,    // [3]
    const unsigned int* n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int N = n[0];
    if (i >= N) return;

    int k = 3 * i;
    double mi = m[i];

    force[k+0] += mi * g[0];
    force[k+1] += mi * g[1];
    force[k+2] += mi * g[2];
}



extern "C" __global__
void dem_full_allpairs(
    const double* x,        // [N*3]
    const double* u,        // [N*3]
    double* force,          // [N*3]
    const double* m,        // [N]
    const double* rad,      // [N]
    const unsigned int* n,
    const double* kn,
    const double* cor_pp,
    const double* friction_pp
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int N = n[0];
    if (i >= N) return;

    int ki = 3 * i;

    // cache xi
    double xi0 = x[ki+0];
    double xi1 = x[ki+1];
    double xi2 = x[ki+2];

    double ui0 = u[ki+0];
    double ui1 = u[ki+1];
    double ui2 = u[ki+2];

    double mi = m[i];
    double ri = rad[i];

    for (int j = 0; j < N; ++j) {
        if (i == j) continue;

        int kj = 3 * j;

        // relative position
        double dx = xi0 - x[kj+0];
        double dy = xi1 - x[kj+1];
        double dz = xi2 - x[kj+2];

        double r2 = dx*dx + dy*dy + dz*dz;
        if (r2 < 1e-24) continue;

        double rij = sqrt(r2);
        double overlap = ri + rad[j] - rij;
        if (overlap <= 0.0) continue;

        double inv_r = 1.0 / rij;
        double nij0 = dx * inv_r;
        double nij1 = dy * inv_r;
        double nij2 = dz * inv_r;

        // relative velocity
        double dvx = ui0 - u[kj+0];
        double dvy = ui1 - u[kj+1];
        double dvz = ui2 - u[kj+2];

        double vij_dot_n = dvx*nij0 + dvy*nij1 + dvz*nij2;

        double vn0 = vij_dot_n * nij0;
        double vn1 = vij_dot_n * nij1;
        double vn2 = vij_dot_n * nij2;

        // ---- safe effective mass ----
        double mj = m[j];
        double m_eff;
        if (mi <= 0.0 && mj <= 0.0) continue;
        else if (mi <= 0.0) m_eff = mj;
        else if (mj <= 0.0) m_eff = mi;
        else m_eff = (mi * mj) / (mi + mj);

        // ---- restitution damping ----
        double e = fmax(cor_pp[0], 1e-6);
        double loge = log(e);
        double beta = loge / sqrt(loge*loge + M_PI*M_PI);
        double eta_n = -2.0 * beta * sqrt(m_eff * kn[0]);

        double fn_spring = kn[0] * overlap;

        // ---- normal force ----
        double fnx = fn_spring * nij0 - eta_n * vn0;
        double fny = fn_spring * nij1 - eta_n * vn1;
        double fnz = fn_spring * nij2 - eta_n * vn2;

        // accumulate
        force[ki+0] += fnx;
        force[ki+1] += fny;
        force[ki+2] += fnz;
    }
}


extern "C" __global__
void freeze_boundary_particles(
                               double* force,      // [N*3]
                               const double* body_type,      // [N]
                               const unsigned int* n
                               ) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int N = n[0];
    if (i >= N) return;

    int k = 3 * i;

    if (body_type[i] == 0.) {
    force[k+0] = 0.;
    force[k+1] = 0.;
    force[k+2] = 0.;
    }
}

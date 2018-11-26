// external crate imports
use rayon::prelude::*;

// local library imports
use super::DEM;
use contact_search::{get_neighbours, NNPS};
use RK2Integrator;

pub fn contact_force_par(
    d_x: &[f32],
    d_y: &[f32],
    d_r: &[f32],
    d_fx: &mut [f32],
    d_fy: &mut [f32],
    s_x: &[f32],
    s_y: &[f32],
    s_r: &[f32],
    s_nnps_id: usize,
    nnps: &NNPS,
    kn: f32,
) {
    d_fx.par_iter_mut()
        .zip(
            d_fy.par_iter_mut()
                .zip(d_x.par_iter().zip(d_y.par_iter().zip(d_r.par_iter()))),
        )
        .for_each(|(d_fx_i, (d_fy_i, (d_x_i, (d_y_i, d_r_i))))| {
            let nbrs = get_neighbours(*d_x_i, *d_y_i, s_nnps_id, &nnps);
            for &j in nbrs.iter() {
                let dx = d_x_i - s_x[j];
                let dy = d_y_i - s_y[j];
                let rij = (dx.powf(2.) + dy.powf(2.)).sqrt();
                // eliminate the interaction between same particle
                if rij > 1e-12 {
                    // if the two particles are in overlap
                    let nij_x = dx / rij;
                    let nij_y = dy / rij;

                    let overlap = d_r_i + s_r[j] - rij;
                    if overlap > 0. {
                        *d_fx_i += kn * overlap * nij_x;
                        *d_fy_i += kn * overlap * nij_y;
                    }
                }
            }
        });
}

pub fn body_force(d_fx: &mut [f32], d_fy: &mut [f32], d_m: &[f32], gx: f32, gy: f32) {
    for i in 0..d_fx.len() {
        d_fx[i] = gx * d_m[i];
        d_fy[i] = gy * d_m[i];
    }
}

impl RK2Integrator for DEM {
    fn initialize(&mut self) {
        let (d_x, d_y, d_u, d_v, d_x0, d_y0, d_u0, d_v0) = (
            &self.x,
            &self.y,
            &self.u,
            &self.v,
            &mut self.x0,
            &mut self.y0,
            &mut self.u0,
            &mut self.v0,
        );
        for i in 0..d_x.len() {
            d_x0[i] = d_x[i];
            d_y0[i] = d_y[i];
            d_u0[i] = d_u[i];
            d_v0[i] = d_v[i];
        }
    }

    fn stage1(&mut self, dt: f32) {
        let dtb2 = dt / 2.;
        let (d_x, d_y, d_u, d_v, d_fx, d_fy, d_x0, d_y0, d_u0, d_v0, d_m) = (
            &mut self.x,
            &mut self.y,
            &mut self.u,
            &mut self.v,
            &self.fx,
            &self.fy,
            &self.x0,
            &self.y0,
            &self.u0,
            &self.v0,
            &self.m,
        );
        let dtb2 = dt / 2.;
        for i in 0..d_x.len() {
            d_u[i] = d_u0[i] + d_fx[i] / d_m[i] * dtb2;
            d_v[i] = d_v0[i] + d_fy[i] / d_m[i] * dtb2;
            d_x[i] = d_x0[i] + d_u[i] * dt;
            d_y[i] = d_y0[i] + d_v[i] * dt;
        }
    }
    fn stage2(&mut self, dt: f32) {
        let (d_x, d_y, d_u, d_v, d_fx, d_fy, d_x0, d_y0, d_u0, d_v0, d_m) = (
            &mut self.x,
            &mut self.y,
            &mut self.u,
            &mut self.v,
            &self.fx,
            &self.fy,
            &self.x0,
            &self.y0,
            &self.u0,
            &self.v0,
            &self.m,
        );

        for i in 0..d_x.len() {
            d_u[i] = d_u0[i] + d_fx[i] / d_m[i] * dt;
            d_v[i] = d_v0[i] + d_fy[i] / d_m[i] * dt;
            d_x[i] = d_x0[i] + d_u[i] * dt;
            d_y[i] = d_y0[i] + d_v[i] * dt;
        }
    }
}

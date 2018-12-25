// external crate imports
use rayon::prelude::*;

// local library imports
use super::DEM;
use crate::contact_search::{NNPS, get_neighbours_1d, get_neighbours_2d, get_neighbours_3d};
use crate::RK2Integrator;

pub fn contact_force_par(
    d_x: &[f32], d_y: &[f32], d_z: &[f32], d_r: &[f32],
    d_fx: &mut [f32], d_fy: &mut [f32], d_fz: &mut [f32],

    s_x: &[f32], s_y: &[f32], s_z: &[f32], s_r: &[f32],
    s_nnps_id: usize, nnps: &NNPS, kn: f32,
) {
    d_fx.par_iter_mut()
        .zip(d_fy.par_iter_mut().zip(d_fz.par_iter_mut().enumerate()))
        .for_each(|(d_fx_i, (d_fy_i, (i, d_fz_i)))| {
            let nbrs = match nnps.dim {
                1 => get_neighbours_1d(d_x[i], d_y[i], d_z[i], s_nnps_id, &nnps),
                2 => get_neighbours_2d(d_x[i], d_y[i], d_z[i], s_nnps_id, &nnps),
                3 => get_neighbours_3d(d_x[i], d_y[i], d_z[i], s_nnps_id, &nnps),
                _ => panic!("Dimension of wrong"),
            };
            for &j in nbrs.iter() {
                let dx = d_x[i] - s_x[j];
                let dy = d_y[i] - s_y[j];
                let dz = d_z[i] - s_z[j];
                let rij = (dx.powf(2.) + dy.powf(2.) + dz.powf(2.)).sqrt();
                // eliminate the interaction between same particle
                if rij > 1e-12{
                    // if the two particles are in overlap
                    let nij_x = dx / rij;
                    let nij_y = dy / rij;
                    let nij_z = dz / rij;

                    let overlap = d_r[i] + s_r[j] - rij;
                    if overlap > 0. {
                        *d_fx_i += kn * overlap * nij_x;
                        *d_fy_i += kn * overlap * nij_y;
                        *d_fz_i += kn * overlap * nij_z;
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
        for i in 0..self.x.len() {
            self.x0[i] = self.x[i];
            self.y0[i] = self.y[i];
            self.u0[i] = self.u[i];
            self.v0[i] = self.v[i];
        }
    }

    fn stage1(&mut self, dt: f32) {
        let dtb2 = dt / 2.;
        for i in 0..self.x.len() {
            self.u[i] = self.u0[i] + self.fx[i] / self.m[i] * dtb2;
            self.v[i] = self.v0[i] + self.fy[i] / self.m[i] * dtb2;
            self.x[i] = self.x0[i] + self.u[i] * dt;
            self.y[i] = self.y0[i] + self.v[i] * dt;
        }
    }
    fn stage2(&mut self, dt: f32) {
        for i in 0..self.x.len() {
            self.u[i] = self.u0[i] + self.fx[i] / self.m[i] * dt;
            self.v[i] = self.v0[i] + self.fy[i] / self.m[i] * dt;
            self.x[i] = self.x0[i] + self.u[i] * dt;
            self.y[i] = self.y0[i] + self.v[i] * dt;
        }
    }
}

// external crate imports
use rayon::prelude::*;

// local library imports
use super::DEM;
use contact_search::{get_neighbours, NNPS};
use RK2Integrator;

pub fn contact_force_par(
    d_x: &[f32],
    d_y: &[f32],
    d_fx: &mut [f32],
    d_fy: &mut [f32],
    s_x: &[f32],
    s_y: &[f32],
    s_nnps_id: usize,
    nnps: &NNPS,
) {
    d_fx.par_iter_mut()
        .zip(d_fy.par_iter_mut().zip(d_x.par_iter().zip(d_y.par_iter())))
        .for_each(|(d_fxi, (d_fyi, (d_xi, d_yi)))| {
            let nbrs = get_neighbours(*d_xi, *d_yi, s_nnps_id, &nnps);
            for &j in nbrs.iter() {
                let dx = d_xi - s_x[j];
                let dy = d_yi - s_y[j];
                *d_fxi += 1e5 * dx;
                *d_fyi += 1e5 * dy;
            }
        });
}

pub fn body_force(d_fx: &mut [f32], d_fy: &mut [f32], gx: f32, gy: f32) {
    d_fx.par_iter_mut().zip(d_fy.par_iter_mut()).for_each(
        |(d_fx_i, d_fy_i)| {
            *d_fx_i = gx;
            *d_fy_i = gy;
        },
    );
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
        d_x0.par_iter_mut()
            .zip(
                d_y0.par_iter_mut()
                    .zip(d_u0.par_iter_mut().zip(d_v0.par_iter_mut().enumerate())),
            )
            .for_each(|(d_x0_i, (d_y0_i, (d_u0_i, (i, d_v0_i))))| {
                *d_x0_i = d_x[i];
                *d_y0_i = d_y[i];
                *d_u0_i = d_u[i];
                *d_v0_i = d_v[i];
            });
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

        d_x.par_iter_mut()
            .zip(
                d_y.par_iter_mut()
                    .zip(d_u.par_iter_mut().zip(d_v.par_iter_mut().enumerate())),
            )
            .for_each(|(d_x_i, (d_y_i, (d_u_i, (i, d_v_i))))| {
                *d_u_i = d_u0[i] + d_fx[i] / d_m[i] * dtb2;
                *d_v_i = d_v0[i] + d_fy[i] / d_m[i] * dtb2;
                *d_x_i = d_x0[i] + *d_u_i * dtb2;
                *d_y_i = d_y0[i] + *d_v_i * dtb2;
            });
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

        d_x.par_iter_mut()
            .zip(
                d_y.par_iter_mut()
                    .zip(d_u.par_iter_mut().zip(d_v.par_iter_mut().enumerate())),
            )
            .for_each(|(d_x_i, (d_y_i, (d_u_i, (i, d_v_i))))| {
                *d_u_i = d_u0[i] + d_fx[i] / d_m[i] * dt;
                *d_v_i = d_v0[i] + d_fy[i] / d_m[i] * dt;
                *d_x_i = d_x0[i] + *d_u_i * dt;
                *d_y_i = d_y0[i] + *d_v_i * dt;
            });
    }
}

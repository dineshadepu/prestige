// external crate imports
use rayon::prelude::*;

// local library imports
use super::WCSPH;
use crate::contact_search::{get_neighbours, NNPS};
use crate::physics::sph::kernel::Kernel;
use crate::RK2Integrator;

pub fn make_accelerations_zero(d_au: &mut [f32], d_av: &mut [f32], d_arho: &mut [f32]) {
    for i in 0..d_au.len() {
        d_au[i] = 0.;
        d_av[i] = 0.;
        d_arho[i] = 0.;
    }
}

/// Equation of state to compute the pressure from density and speed of sound
/// See pg no 8, eq 3.7 in `Smoothed Particle Hydrodynamics A Study of the
/// possibilities of SPH in hydraulic engineering`
//
// Use this equation in the following way
//
// equation_of_state(&mut d_p, &d_rho, d_rho0, gamma, c);
pub fn equation_of_state(d_p: &mut [f32], d_rho: &[f32], d_rho0: f32, gamma: f32, c: f32) {
    // B = c^2 \rho_0 / gamma
    let b = c.powf(2.) * d_rho0 / gamma;
    for i in 0..d_p.len() {
        d_p[i] = b * ((d_rho[i] / d_rho0).powf(gamma) - 1.);
    }
}

pub fn summation_density(
    d_x: &[f32],
    d_y: &[f32],
    d_h: &[f32],
    d_m: &[f32],
    d_rho: &mut [f32],
    s_x: &[f32],
    s_y: &[f32],
    s_nnps_id: usize,
    nnps: &NNPS,
    kernel: &(dyn Kernel + Sync),
) {
    d_rho.par_iter_mut().enumerate().for_each(|(i, d_rho_i)| {
        let nbrs = get_neighbours(d_x[i], d_y[i], s_nnps_id, &nnps);
        for &j in nbrs.iter() {
            let dx = d_x[i] - s_x[j];
            let dy = d_y[i] - s_y[j];
            let rij = (dx.powf(2.) + dy.powf(2.)).sqrt();
            let wij = kernel.get_wij(rij, d_h[i]);

            *d_rho_i += d_m[i] * wij;
        }
    });
}

/// Continuity equation to compute rate of change of density of a particle.
//
// continuity_equation(&d_x, &d_y, &d_u, &d_v, &d_h, &mut d_arho, &s_x, &s_y,
//                     &s_u, &s_v, &s_m, s_nnps_id, &nnps, &kernel);
pub fn continuity_equation(
    d_x: &[f32],
    d_y: &[f32],
    d_u: &[f32],
    d_v: &[f32],
    d_h: &[f32],
    d_arho: &mut [f32],
    s_x: &[f32],
    s_y: &[f32],
    s_u: &[f32],
    s_v: &[f32],
    s_m: &[f32],
    s_nnps_id: usize,
    nnps: &NNPS,
    kernel: &(dyn Kernel + Sync),
) {
    d_arho.par_iter_mut().enumerate().for_each(|(i, d_arho_i)| {
        // let mut wij = 0.;
        let mut dwij = vec![0., 0.];
        let mut xij = vec![0., 0.];
        let mut uij = vec![0., 0.];
        let mut rij;
        let nbrs = get_neighbours(d_x[i], d_y[i], s_nnps_id, &nnps);
        for &j in nbrs.iter() {
            xij[0] = d_x[i] - s_x[j];
            xij[1] = d_y[i] - s_y[j];
            uij[0] = d_u[i] - s_u[j];
            uij[1] = d_v[i] - s_v[j];
            rij = (xij[0] * xij[0] + xij[1] * xij[1]).sqrt();
            kernel.get_dwij(&xij, &mut dwij, rij, d_h[i]);
            // wij = kernel.get_wij(rij, d_h[i]);

            // computing the rate of change of density of particle i
            *d_arho_i += s_m[j] * (uij[0] * dwij[0] + uij[1] * dwij[1])
        }
    });
}

#[macro_export]
macro_rules! macro_continuity_equation {
    ($func_name:ident, $dest:ident, $source:ident, $nnps:ident, $kernel:ident) => {
        $func_name(
            &$dest.x,
            &$dest.y,
            &$dest.u,
            &$dest.v,
            &$dest.h,
            &mut $dest.arho,
            &$source.x,
            &$source.y,
            &$source.u,
            &$source.v,
            &$source.m,
            $source.nnps_idx,
            &$nnps,
            &$kernel,
        );
    };
}

/// Momentum equation to compute rate of change of velocity of a particle.
//
// momentum_equation(&d_x, &d_y, &d_u, &d_v, &d_h, &d_p, &d_rho, &mut d_au, &mut d_av,
//                   &s_x, &s_y, &s_u, &s_v, &s_m, &s_p, &s_rho, s_nnps_id, &nnps, &kernel);
pub fn momentum_equation(
    d_x: &[f32],
    d_y: &[f32],
    d_u: &[f32],
    d_v: &[f32],
    d_h: &[f32],
    d_p: &[f32],
    d_rho: &[f32],
    d_c: &[f32],
    d_au: &mut [f32],
    d_av: &mut [f32],

    s_x: &[f32],
    s_y: &[f32],
    s_u: &[f32],
    s_v: &[f32],
    s_h: &[f32],
    s_m: &[f32],
    s_p: &[f32],
    s_rho: &[f32],
    s_c: &[f32],
    s_nnps_id: usize,
    alpha: f32,
    nnps: &NNPS,
    kernel: &(dyn Kernel + Sync),
) {
    d_au.par_iter_mut()
        .zip(d_av.par_iter_mut().enumerate())
        .for_each(|(d_au_i, (i, d_av_i))| {
            // let mut wij = 0.;
            // wij = kernel.get_wij(rij, d_h[i]);
            let mut dwij = vec![0., 0.];
            let mut xij = vec![0., 0.];
            let mut uij = vec![0., 0.];
            let mut rij;
            let mut cij;
            let mut hij;
            let mut tmp;
            let mut uij_dot_xij;
            let mut rhoij;
            let mut muij;
            let mut art_vis = 0.;
            let nbrs = get_neighbours(d_x[i], d_y[i], s_nnps_id, &nnps);
            for &j in nbrs.iter() {
                xij[0] = d_x[i] - s_x[j];
                xij[1] = d_y[i] - s_y[j];
                uij[0] = d_u[i] - s_u[j];
                uij[1] = d_v[i] - s_v[j];
                rij = (xij[0].powf(2.) + xij[1].powf(2.)).sqrt();
                kernel.get_dwij(&xij, &mut dwij, rij, d_h[i]);

                // artificial viscosity
                if alpha > 0. {
                    uij_dot_xij = uij[0] * xij[0] + uij[0] * xij[0];
                    cij = 0.5 * (d_c[i] + s_c[j]);
                    hij = 0.5 * (d_h[i] + s_h[j]);

                    rhoij = 0.5 * (d_rho[i] + s_rho[j]);
                    muij = (hij * uij_dot_xij) / (rij.powf(2.) + (0.1 * hij).powf(2.));
                    art_vis = -alpha * cij * muij / rhoij;
                };

                tmp = s_m[j] * (d_p[i] / d_rho[i].powf(2.) + s_p[j] / s_rho[j].powf(2.)) + art_vis;
                *d_au_i -= tmp * dwij[0];
                *d_av_i -= tmp * dwij[1];
            }
        });
}

#[macro_export]
macro_rules! macro_momentum_equation {
    ($func_name:ident, $dest:ident, $source:ident, $nnps:ident, $kernel:ident, $self_src:ident) => {{
        if $self_src == true {
            $func_name(
                &$dest.x,
                &$dest.y,
                &$dest.u,
                &$dest.v,
                &$dest.h,
                &$dest.p,
                &$dest.rho,
                &$dest.c,
                &mut $dest.au,
                &mut $dest.av,
                &$src.x,
                &$src.y,
                &$src.u,
                &$src.v,
                &$src.h,
                &$src.m,
                &$src.p,
                &$src.rho,
                &$src.c,
                $src.nnps_idx,
                &$nnps,
                &$kernel,
            );
        }
    }};
}

impl RK2Integrator for WCSPH {
    fn initialize(&mut self) {
        for i in 0..self.x.len() {
            self.x0[i] = self.x[i];
            self.y0[i] = self.y[i];
            self.u0[i] = self.u[i];
            self.v0[i] = self.v[i];
            self.rho0[i] = self.rho[i];
        }
    }

    fn stage1(&mut self, dt: f32) {
        let dtb2 = dt / 2.;
        for i in 0..self.x.len() {
            self.rho[i] = self.rho0[i] + self.arho[i] * dtb2;
            self.u[i] = self.u0[i] + self.au[i] * dtb2;
            self.v[i] = self.v0[i] + self.av[i] * dtb2;
            self.x[i] = self.x0[i] + self.u[i] * dt;
            self.y[i] = self.y0[i] + self.v[i] * dt;
        }
    }
    fn stage2(&mut self, dt: f32) {
        for i in 0..self.x.len() {
            self.rho[i] = self.rho0[i] + self.arho[i] * dt;
            self.u[i] = self.u0[i] + self.au[i] * dt;
            self.v[i] = self.v0[i] + self.av[i] * dt;
            self.x[i] = self.x0[i] + self.u[i] * dt;
            self.y[i] = self.y0[i] + self.v[i] * dt;
        }
    }
}

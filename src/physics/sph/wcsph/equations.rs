// external crate imports
use rayon::prelude::*;

// local library imports
use super::WCSPH;
use crate::contact_search::{NNPSGeneric};
use crate::physics::sph::kernel::Kernel;
use crate::{EulerIntegrator, RK2Integrator};


pub fn reset_wcsph_entity(entity: &mut WCSPH){
    for i in 0..entity.x.len(){
        entity.arho[i] = 0.;
        entity.au[i] = 0.;
        entity.av[i] = 0.;
        entity.aw[i] = 0.;
        entity.ax[i] = 0.;
        entity.ay[i] = 0.;
        entity.az[i] = 0.;
    }
}

pub fn apply_gravity(
    d_au: &mut [f32],
    d_av: &mut [f32],
    d_aw: &mut [f32],
    gx: f32,
    gy: f32,
    gz: f32,
) {
    for i in 0..d_au.len() {
        d_au[i] += gx;
        d_av[i] += gy;
        d_aw[i] += gz;
    }
}

#[macro_export]
macro_rules! apply_gravity_macro{
    ($dest:ident, $gx:expr, $gy:expr, $gz:expr) => {
        apply_gravity(
            &mut $dest.au, &mut $dest.av, &mut $dest.aw,
        );
    };
}

/// Equation of state to compute the pressure from density and speed of sound
/// See pg no 8, eq 3.7 in `Smoothed Particle Hydrodynamics A Study of the
/// possibilities of SPH in hydraulic engineering`
//
// Use this equation in the following way
//
// equation_of_state(&mut d_p, &d_rho, rho_rest, gamma, c);
pub fn tait_equation(d_p: &mut [f32], d_cs: &mut [f32], d_rho: &[f32], rho0: f32,
                     gamma: f32, c0: f32) {
    let frac_1_rho0 = 1. / rho0;
    let b = rho0 * c0 * c0 / gamma;
    let gamma1 = 0.5 * (gamma - 1.);

    for i in 0..d_p.len() {
        let ratio = d_rho[i] * frac_1_rho0;

        d_p[i] = b * (ratio.powf(gamma) - 1.);
        d_cs[i] = c0 * ratio.powf(gamma1);
    }
}


pub fn tait_hgcorrection_equation(d_p: &mut [f32], d_cs: &mut [f32], d_rho: &mut [f32], rho0: f32,
                                  gamma: f32, c0: f32) {
    let frac_1_rho0 = 1. / rho0;
    let b = rho0 * c0 * c0 / gamma;
    let gamma1 = 0.5 * (gamma - 1.);

    for i in 0..d_p.len() {
        if d_rho[i] < rho0 {
            d_rho[i] = rho0;
        }
        let ratio = d_rho[i] * frac_1_rho0;

        d_p[i] = b * (ratio.powf(gamma) - 1.);
        d_cs[i] = c0 * ratio.powf(gamma1);
    }
}

pub fn summation_density(
    d_x: &[f32], d_y: &[f32], d_z: &[f32], d_h: &[f32],
    d_m: &[f32], d_rho: &mut [f32], s_x: &[f32], s_y: &[f32],
    s_z: &[f32], s_nnps_id: usize, nnps: &(dyn NNPSGeneric+ Sync),
    kernel: &(dyn Kernel + Sync))
{
    d_rho.par_iter_mut().enumerate().for_each(|(i, d_rho_i)| {
        let nbrs = nnps.get_neighbours(d_x[i], d_y[i], d_z[i], s_nnps_id);
        for &j in nbrs.iter() {
            let dx = d_x[i] - s_x[j];
            let dy = d_y[i] - s_y[j];
            let dz = d_z[i] - s_z[j];
            let rij = (dx.powf(2.) + dy.powf(2.) + dz.powf(2.)).sqrt();
            let wij = kernel.get_wij(rij, d_h[i]);

            *d_rho_i += d_m[i] * wij;
        }
    });
}

/// Continuity equation to compute rate of change of density of a particle.
//
pub fn continuity_equation(
    d_x: &[f32], d_y: &[f32], d_z: &[f32], d_u: &[f32],
    d_v: &[f32], d_w: &[f32], d_h: &[f32], d_arho: &mut [f32],
    s_x: &[f32], s_y: &[f32], s_z: &[f32], s_u: &[f32],
    s_v: &[f32], s_w: &[f32], s_m: &[f32], s_nnps_id: usize,
    nnps: &(dyn NNPSGeneric + Sync), kernel: &(dyn Kernel + Sync),)
{
    d_arho.par_iter_mut().enumerate().for_each(|(i, d_arho_i)| {
        // let mut wij = 0.;
        let mut dwij = vec![0.; 3];
        let mut xij = vec![0.; 3];
        let mut uij = vec![0.; 3];
        let mut rij;
        let nbrs = nnps.get_neighbours(d_x[i], d_y[i], d_z[i], s_nnps_id);
        for &j in nbrs.iter() {
            xij[0] = d_x[i] - s_x[j];
            xij[1] = d_y[i] - s_y[j];
            xij[2] = d_z[i] - s_z[j];
            uij[0] = d_u[i] - s_u[j];
            uij[1] = d_v[i] - s_v[j];
            uij[2] = d_w[i] - s_w[j];
            rij = (xij[0] * xij[0] + xij[1] * xij[1] + xij[2] * xij[2]).sqrt();
            kernel.get_dwij(&xij, &mut dwij, rij, d_h[i]);

            // computing the rate of change of density of particle i
            *d_arho_i += s_m[j] * (uij[0] * dwij[0] + uij[1] * dwij[1] + uij[2] * dwij[2]);
        }
    });
}

#[macro_export]
macro_rules! continuity_eq_macro {
    ($dest:ident, ($($sources:ident),*), $nnps:ident, $kernel:ident) => {
        $(continuity_equation(
            &$dest.x, &$dest.y, &$dest.z, &$dest.u,
            &$dest.v, &$dest.w, &$dest.h, &mut $dest.arho,

            &$sources.x, &$sources.y, &$sources.z, &$sources.u,
            &$sources.v, &$sources.w, &$sources.m, $sources.nnps_idx,

            &$nnps, &$kernel,
        );)*
    };
}

/// Momentum equation to compute rate of change of velocity of a particle.
/// https://pdfs.semanticscholar.org/50d1/76a68ea2088d6256ef192a5fdf7a27f41f5e.pdf
pub fn momentum_equation(
    d_x: &[f32], d_y: &[f32], d_z: &[f32], d_u: &[f32], d_v: &[f32],
    d_w: &[f32], d_h: &[f32], d_p: &[f32], d_rho: &[f32], d_cs: &[f32],
    d_au: &mut [f32], d_av: &mut [f32], d_aw: &mut [f32],

    s_x: &[f32], s_y: &[f32], s_z: &[f32], s_u: &[f32],
    s_v: &[f32], s_w: &[f32], s_h: &[f32], s_m: &[f32],
    s_p: &[f32], s_rho: &[f32], s_cs: &[f32], s_nnps_id: usize,

    alpha: f32, beta: f32,
    nnps: &(dyn NNPSGeneric + Sync), kernel: &(dyn Kernel + Sync),)
{
    d_au.par_iter_mut()
        .zip(d_av.par_iter_mut()
             .zip(d_aw.par_iter_mut().enumerate()))
        .for_each(|(d_au_i, (d_av_i, (i, d_aw_i)))| {
            let mut dwij = vec![0.; 3];
            let mut xij = vec![0.; 3];
            let mut uij = vec![0.; 3];
            let (
                mut rij,
                mut rij_2,
                mut hij,
                mut tmp,
                mut uij_dot_xij,
                mut muij,
                mut piij,
                mut cij,
                mut tmpi,
                mut tmpj,
                mut frac_1_rhoij,
            );
            let nbrs = nnps.get_neighbours(d_x[i], d_y[i], d_z[i], s_nnps_id);
            for &j in nbrs.iter() {
                // common code
                xij[0] = d_x[i] - s_x[j];
                xij[1] = d_y[i] - s_y[j];
                xij[2] = d_z[i] - s_z[j];
                uij[0] = d_u[i] - s_u[j];
                uij[1] = d_v[i] - s_v[j];
                uij[2] = d_w[i] - s_w[j];
                rij = (xij[0] * xij[0] + xij[1] * xij[1] + xij[2] * xij[2]).sqrt();
                rij_2 = rij * rij;
                kernel.get_dwij(&xij, &mut dwij, rij, d_h[i]);
                // common code

                let frac_1_rhoi2 = 1. / (d_rho[i] * d_rho[i]);
                let frac_1_rhoj2 = 1. / (s_rho[j] * s_rho[j]);

                uij_dot_xij = uij[0] * xij[0] + uij[1] * xij[1] + uij[2] * xij[2];
                piij = 0.0;

                // artificial viscosity
                if uij_dot_xij < 0. {
                    hij = 0.5 * (d_h[i] + s_h[j]);
                    cij = 0.5 * (d_cs[i] + s_cs[j]);
                    frac_1_rhoij = 1. / (0.5 * (d_rho[i] + s_rho[j]));

                    muij = (hij * uij_dot_xij)/(rij_2 + 1e-12);

                    piij = -alpha*cij*muij + beta*muij*muij;
                    piij = piij*frac_1_rhoij;
                }

                tmpi = d_p[i] * frac_1_rhoi2;
                tmpj = s_p[j] * frac_1_rhoj2;

                tmp = tmpi + tmpj;

                *d_au_i += -s_m[j] * (tmp + piij) * dwij[0];
                *d_av_i += -s_m[j] * (tmp + piij) * dwij[1];
                *d_aw_i += -s_m[j] * (tmp + piij) * dwij[2];
            }
        });
}

#[macro_export]
macro_rules! momentum_eq_macro {
    ($dest:ident, ($($source:ident,)*), $nnps:ident, $kernel:ident, $alpha:expr, $beta:expr) => {
        $(momentum_equation(
            &$dest.x, &$dest.y, &$dest.z, &$dest.u, &$dest.v,
            &$dest.w, &$dest.h, &$dest.p, &$dest.rho, &$dest.cs,
            &mut $dest.au, &mut $dest.av, &mut $dest.aw,

            &$sources.x, &$sources.y, &$sources.z, &$sources.u,
            &$sources.v, &$sources.w, &$sources.h, &$sources.m,
            &$sources.p, &$sources.rho, &$sources.cs, $sources.nnps_idx,

            $alpha, $beta, &$nnps, &$kernel
        );)*
    };
}

pub fn continuity_and_momentum_equation(
    d_x: &[f32], d_y: &[f32], d_z: &[f32], d_u: &[f32], d_v: &[f32],
    d_w: &[f32], d_h: &[f32], d_p: &[f32], d_rho: &[f32],
    d_cs: &[f32], d_arho: &mut [f32],  d_au: &mut [f32], d_av: &mut [f32],
    d_aw: &mut [f32],

    s_x: &[f32], s_y: &[f32], s_z: &[f32], s_u: &[f32],
    s_v: &[f32], s_w: &[f32], s_h: &[f32], s_m: &[f32],
    s_p: &[f32], s_rho: &[f32], s_cs: &[f32], s_nnps_id: usize,

    alpha: f32, beta: f32,
    nnps: &(dyn NNPSGeneric + Sync), kernel: &(dyn Kernel + Sync),)
{
    d_arho.par_iter_mut()
        .zip(d_au.par_iter_mut().zip(d_av.par_iter_mut().zip(d_aw.par_iter_mut().enumerate())))
        .for_each(|(d_arho_i, (d_au_i, (d_av_i, (i, d_aw_i))))| {
            let mut dwij = vec![0.; 3];
            let mut xij = vec![0.; 3];
            let mut uij = vec![0.; 3];
            let (
                mut rij,
                mut rij_2,
                mut hij,
                mut tmp,
                mut uij_dot_xij,
                mut muij,
                mut piij,
                mut cij,
                mut tmpi,
                mut tmpj,
                mut frac_1_rhoij,
            );
            let nbrs = nnps.get_neighbours(d_x[i], d_y[i], d_z[i], s_nnps_id);
            for &j in nbrs.iter() {
                xij[0] = d_x[i] - s_x[j];
                xij[1] = d_y[i] - s_y[j];
                xij[2] = d_z[i] - s_z[j];
                uij[0] = d_u[i] - s_u[j];
                uij[1] = d_v[i] - s_v[j];
                uij[2] = d_w[i] - s_w[j];
                rij = (xij[0] * xij[0] + xij[1] * xij[1] + xij[2] * xij[2]).sqrt();
                rij_2 = rij * rij;
                kernel.get_dwij(&xij, &mut dwij, rij, d_h[i]);

                // continuity equation
                *d_arho_i += s_m[j] * (uij[0] * dwij[0] + uij[1] * dwij[1] + uij[2] * dwij[2]);

                let frac_1_rhoi2 = 1. / (d_rho[i] * d_rho[i]);
                let frac_1_rhoj2 = 1. / (s_rho[j] * s_rho[j]);

                uij_dot_xij = uij[0] * xij[0] + uij[1] * xij[1] + uij[2] * xij[2];
                piij = 0.0;

                // artificial viscosity
                if uij_dot_xij < 0. {
                    hij = 0.5 * (d_h[i] + s_h[j]);
                    cij = 0.5 * (d_cs[i] + s_cs[j]);
                    frac_1_rhoij = 1. / (0.5 * (d_rho[i] + s_rho[j]));

                    muij = (hij * uij_dot_xij)/(rij_2 + 1e-12);

                    piij = -alpha*cij*muij + beta*muij*muij;
                    piij = piij*frac_1_rhoij;
                }

                tmpi = d_p[i] * frac_1_rhoi2;
                tmpj = s_p[j] * frac_1_rhoj2;

                tmp = tmpi + tmpj;

                *d_au_i += -s_m[j] * (tmp + piij) * dwij[0];
                *d_av_i += -s_m[j] * (tmp + piij) * dwij[1];
                *d_aw_i += -s_m[j] * (tmp + piij) * dwij[2];
            }
        });
}


#[macro_export]
macro_rules! continuity_and_momentum_eq_macro {
    ($dest:ident, ($($sources:ident),*), $nnps:ident, $kernel:ident, $alpha:expr, $beta:expr) => {
        $(continuity_and_momentum_equation(
            &$dest.x, &$dest.y, &$dest.z, &$dest.u, &$dest.v,
            &$dest.w, &$dest.h, &$dest.p, &$dest.rho,
            &$dest.cs,
            &mut $dest.arho, &mut $dest.au, &mut $dest.av, &mut $dest.aw,

            &$sources.x, &$sources.y, &$sources.z, &$sources.u, &$sources.v,
            &$sources.w, &$sources.h, &$sources.m, &$sources.p, &$sources.rho,
            &$sources.cs, $sources.nnps_idx,

            $alpha, $beta, &$nnps, &$kernel,
        );)*
    };
}

pub fn xsph_equation(
    d_x: &[f32], d_y: &[f32], d_z: &[f32], d_u: &[f32],
    d_v: &[f32], d_w: &[f32], d_h: &[f32], d_rho: &[f32],
    d_ax: &mut[f32], d_ay: &mut[f32], d_az: &mut[f32],
    s_x: &[f32], s_y: &[f32], s_z: &[f32], s_u: &[f32],
    s_v: &[f32], s_w: &[f32], s_m: &[f32], s_rho: &[f32], s_nnps_id: usize,
    eps: f32,
    nnps: &(dyn NNPSGeneric + Sync), kernel: &(dyn Kernel + Sync),)
{
    d_ax.par_iter_mut()
        .zip(d_ay.par_iter_mut()
             .zip(d_az.par_iter_mut().enumerate()))
        .for_each(|(d_ax_i, (d_ay_i, (i, d_az_i)))| {
            let mut dwij = vec![0.; 3];
            let mut xij = vec![0.; 3];
            let mut uij = vec![0.; 3];
            // let mut rij: f32;
            let (
                mut rij,
                mut tmp,
                mut frac_1_rhoij,
                mut wij,
            );
            let nbrs = nnps.get_neighbours(d_x[i], d_y[i], d_z[i], s_nnps_id);
            for &j in nbrs.iter() {
                // common code
                xij[0] = d_x[i] - s_x[j];
                xij[1] = d_y[i] - s_y[j];
                xij[2] = d_z[i] - s_z[j];
                uij[0] = d_u[i] - s_u[j];
                uij[1] = d_v[i] - s_v[j];
                uij[2] = d_w[i] - s_w[j];
                rij = (xij[0] * xij[0] + xij[1] * xij[1] + xij[2] * xij[2]).sqrt();
                kernel.get_dwij(&xij, &mut dwij, rij, d_h[i]);
                wij = kernel.get_wij(rij, d_h[i]);
                frac_1_rhoij = 1. / (0.5 * (d_rho[i] + s_rho[j]));
                // common code
                tmp = -eps * s_m[j] * wij * frac_1_rhoij;
                *d_ax_i += tmp * uij[0];
                *d_ay_i += tmp * uij[1];
                *d_az_i += tmp * uij[2];
            }
        });

    for i in 0..d_ax.len(){
        d_ax[i] += d_u[i];
        d_ay[i] += d_v[i];
        d_az[i] += d_w[i];
    }
}


#[macro_export]
macro_rules! xsph_macro {
    ($dest:ident, ($($sources:ident),*), $nnps:ident, $kernel:ident, $eps:expr) => {
        $(xsph_equation(
            &$dest.x, &$dest.y, &$dest.z, &$dest.u,
            &$dest.v, &$dest.w, &$dest.h, &$dest.rho,
            &mut $dest.ax, &mut $dest.ay, &mut $dest.az,
            &$sources.x, &$sources.y, &$sources.z, &$sources.u,
            &$sources.v, &$sources.w, &$sources.m, &$sources.rho,
            $sources.nnps_idx,

            $eps, &$nnps, &$kernel,
        );)*
    };
}


impl RK2Integrator for WCSPH {
    fn rk2_initialize(&mut self) {
        for i in 0..self.x.len() {
            self.x0[i] = self.x[i];
            self.y0[i] = self.y[i];
            self.z0[i] = self.z[i];
            self.u0[i] = self.u[i];
            self.v0[i] = self.v[i];
            self.w0[i] = self.w[i];
            self.rho0[i] = self.rho[i];
        }
    }

    fn rk2_stage_1(&mut self, dt: f32) {
        let dtb2 = dt / 2.;
        for i in 0..self.x.len() {
            self.rho[i] = self.rho0[i] + self.arho[i] * dtb2;
            self.u[i] = self.u0[i] + self.au[i] * dtb2;
            self.v[i] = self.v0[i] + self.av[i] * dtb2;
            self.w[i] = self.w0[i] + self.aw[i] * dtb2;
            self.x[i] = self.x0[i] + self.u[i] * dtb2;
            self.y[i] = self.y0[i] + self.v[i] * dtb2;
            self.z[i] = self.z0[i] + self.w[i] * dtb2;
        }
    }
    fn rk2_stage_2(&mut self, dt: f32) {
        for i in 0..self.x.len() {
            self.rho[i] = self.rho0[i] + self.arho[i] * dt;
            self.u[i] = self.u0[i] + self.au[i] * dt;
            self.v[i] = self.v0[i] + self.av[i] * dt;
            self.w[i] = self.w0[i] + self.aw[i] * dt;
            self.x[i] = self.x0[i] + self.u[i] * dt;
            self.y[i] = self.y0[i] + self.v[i] * dt;
            self.z[i] = self.z0[i] + self.w[i] * dt;
        }
    }
}

impl EulerIntegrator for WCSPH{
    fn euler_stage_1(&mut self, dt: f32){
        for i in 0..self.x.len(){
            self.rho[i] = self.rho[i] + self.arho[i] * dt;
            self.u[i] += self.au[i]*dt;
            self.v[i] += self.av[i]*dt;
            self.w[i] += self.aw[i]*dt;

            self.x[i] += self.u[i]*dt;
            self.y[i] += self.v[i]*dt;
            self.z[i] += self.w[i]*dt;
        }
    }
}

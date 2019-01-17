use super::{DEMLinear, TangCt};
use crate::contact_search::{NNPSGeneric};
use crate::EulerIntegrator;
use cgmath::prelude::*;
use cgmath::{Vector3, Matrix3};
use rayon::prelude::*;
use std::collections::HashMap;

pub fn linear_dem_interparticle_force(
    d_x: &[f32], d_y: &[f32], d_z: &[f32], d_u: &[f32],
    d_v: &[f32], d_w: &[f32], d_omega_x: &[f32], d_omega_y: &[f32],
    d_omega_z: &[f32], d_rad: &[f32], d_fx: &mut [f32], d_fy: &mut [f32],
    d_fz: &mut [f32], d_tx: &mut [f32], d_ty: &mut [f32], d_tz: &mut [f32],
    d_tng_ctcs: &mut [Vec<HashMap<usize, TangCt>>],

    s_x: &[f32], s_y: &[f32], s_z: &[f32], s_u: &[f32],
    s_v: &[f32], s_w: &[f32], s_omega_x: &[f32],
    s_omega_y: &[f32], s_omega_z: &[f32], s_dem_id:usize,
    s_nnps_id: usize,

    nnps: &(dyn NNPSGeneric + Sync),
    k_n: f32,
    eta_n: f32,
    k_t: f32,
    eta_t: f32,
    mu_f: f32,
    dt: f32,

) {
    d_fx.par_iter_mut()
        .zip(
            d_fy.par_iter_mut().zip(
                d_fz.par_iter_mut().zip(
                    d_tx.par_iter_mut().zip(
                        d_ty.par_iter_mut().zip(
                            d_tz.par_iter_mut().zip(
                                d_tng_ctcs.par_iter_mut().enumerate()))))))
        .for_each(|(d_fx_i, (d_fy_i, (d_fz_i, (d_tx_i, (d_ty_i, (d_tz_i, (i, d_tng_ctcs_i)))))))| {
            let mut xij = Vector3::new(0., 0., 0.);
            let mut vij = Vector3::new(0., 0., 0.);
            let mut nij;
            let mut tij;
            let mut fij_n = Vector3::new(0., 0., 0.);
            let mut fij_t = Vector3::new(0., 0., 0.);
            let mut torque;
            let omega_i = Vector3::new(d_omega_x[i], d_omega_y[i], d_omega_z[i]);
            let mut omega_j;
            let mut vij_n;
            let mut vij_t = Vector3::new(0., 0., 0.);
            let mut tmp;
            let mut h_v;
            let mut h_v_norm;
            let mut rot_mtrx = Matrix3::<f32>::zero();
            let tc_i_with_s_dem = &mut d_tng_ctcs_i[s_dem_id];

            let mut rij;
            let mut overlap_n;
            let mut couloumb_limit;
            let mut angle;
            let mut q;
            let mut s;
            let mut c;
            // let mut vij_dot_xij = 0.;
            let nbrs = nnps.get_neighbours(d_x[i], d_y[i], d_z[i], s_nnps_id);

            for &j in nbrs.iter() {
                // Reset the forces for next contact
                fij_n[0] = 0.;
                fij_n[1] = 0.;
                fij_n[2] = 0.;
                fij_t[0] = 0.;
                fij_t[1] = 0.;
                fij_t[2] = 0.;

                xij[0] = s_x[j] - d_x[i];
                xij[1] = s_y[j] - d_y[i];
                xij[2] = s_z[j] - d_z[i];
                rij = xij.magnitude();

                // eliminate self interaction
                if rij > 1e-12 {
                    // overlap amount
                    overlap_n = d_rad[i] + d_rad[j] - rij;
                    if overlap_n > 0. {
                        omega_j = Vector3::new(s_omega_x[i], s_omega_y[i], s_omega_z[i]);
                        // normal vector from i to j
                        // Be careful about this
                        nij = xij / rij;

                        // relative velocity at the surface due to angular velocity
                        tmp = (d_rad[i] * omega_i + d_rad[j] * omega_j).cross(nij);

                        // Relative velocity particle i w.r.t j is uij
                        vij[0] = d_u[i] - s_u[j] + tmp[0];
                        vij[1] = d_v[i] - s_v[j] + tmp[1];
                        vij[2] = d_w[i] - s_w[j] + tmp[2];

                        // normal velocity is
                        vij_n = vij.dot(nij) * nij;

                        // --------------------------------
                        // normal force due to the normal overlap
                        fij_n = -k_n * overlap_n * nij - eta_n * vij_n;

                        // --------------------------------
                        // Compute tangential force
                        // check if the particle j is in contact history
                        if let Some(tng_ct) = tc_i_with_s_dem.get_mut(&j) {
                            // since we are tracking the particle, we will have
                            // the information about the tangential overlap of
                            // particle j with i. It will be at
                            // tng_ct
                            // Rotate the tangential overlap
                            // First compute the rotation matrix
                            // Let vector about which we rotate be rot_vec
                            h_v = nij.cross(tng_ct.normal_0);
                            h_v_norm = h_v / h_v.magnitude();
                            angle = h_v.magnitude().asin();
                            c = angle.cos();
                            s = angle.sin();
                            q = 1. - c;

                            // rot_mtrx is a column major matrix. Used from cgmath
                            // Column 0 row 0 term
                            rot_mtrx.x[0] = q * h_v_norm.x.powf(2.) + c;
                            rot_mtrx.x[1] = q * h_v_norm.x * h_v_norm.y + s * h_v_norm.z;
                            rot_mtrx.x[2] = q * h_v_norm.x * h_v_norm.z - s * h_v_norm.y;

                            rot_mtrx.y[0] = q * h_v_norm.x * h_v_norm.y - s * h_v_norm.z;
                            rot_mtrx.y[1] = q * h_v_norm.y.powf(2.) + c;
                            rot_mtrx.y[2] = q * h_v_norm.y * h_v_norm.z + s * h_v_norm.x;

                            rot_mtrx.z[0] = q * h_v_norm.x * h_v_norm.z + s * h_v_norm.y;
                            rot_mtrx.z[1] = q * h_v_norm.y * h_v_norm.z - s * h_v_norm.x;
                            rot_mtrx.z[2] = q * h_v_norm.z.powf(2.) + c;

                            // Rotate the tangential overlap using the matrix
                            tng_ct.delta_tng = rot_mtrx * tng_ct.delta_tng;

                            // Velocity in tangential direction for tangential damping force
                            vij_t = vij - vij_n;

                            // use the rotated delta_tng for force computation
                            fij_t = -k_t * tng_ct.delta_tng - eta_t * vij_t;

                            // Compare it against the Coulomb limit friction
                            couloumb_limit = mu_f * fij_n.magnitude();

                            if fij_t.magnitude() > couloumb_limit {
                                tij = vij_t / vij_t.magnitude();
                                fij_t = -couloumb_limit * tij;
                                // limit the tangential overlap for maximum friction
                                // force and update it to next time step
                                tng_ct.delta_tng = couloumb_limit / k_t * tij + vij_t * dt;
                            } else {
                                // If the force is less than coulomb limit then, simply update
                                // it to next time step
                                tng_ct.delta_tng += vij_t * dt;
                            }
                            // update the normal vector to current normal vector
                            tng_ct.normal_0 = nij;
                        } else {
                            // Since it is not been tracked, we need to to add the particle
                            // Since it is the first time contact we will not compute
                            // any tangential force. We will increment the tangential overlap
                            // to the next time step and save it as a new contact.
                            tc_i_with_s_dem.insert(j, TangCt::new(&nij, vij_t*dt));
                        }

                        *d_fx_i += fij_n[0] + fij_t[0];
                        *d_fy_i += fij_n[1] + fij_t[1];
                        *d_fz_i += fij_n[2] + fij_t[2];

                        // compute torque
                        // torque is
                        torque = d_rad[i] * nij.cross(fij_t);
                        *d_tx_i += torque[0];
                        *d_ty_i += torque[1];
                        *d_tz_i += torque[2];


                    } else {
                        // This condition implies that the particles are not in
                        // contact, so remove the particle j from the contact
                        // list of particle i if it is been tracked
                        tc_i_with_s_dem.remove(&j);
                    }
                }
            }
        });
}

impl EulerIntegrator for DEMLinear{
    fn euler_stage_1(&mut self, dt: f32){
        for i in 0..self.x.len(){
            self.u[i] += self.fx[i]*dt;
            self.v[i] += self.fy[i]*dt;
            self.w[i] += self.fz[i]*dt;

            self.x[i] += self.u[i]*dt;
            self.y[i] += self.v[i]*dt;
            self.z[i] += self.w[i]*dt;
        }
    }
}

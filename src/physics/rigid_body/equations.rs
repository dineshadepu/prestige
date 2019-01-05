use super::RB3d;
use crate::contact_search::{NNPS, get_neighbours_1d, get_neighbours_2d, get_neighbours_3d};
use crate::EulerIntegrator;
use cgmath::prelude::*;
use cgmath::Matrix3;
use cgmath::Vector3;
use rayon::prelude::*;

impl RB3d {
    /// compute total mass
    /// *Note*: This will be called while computing center of mass
    pub fn compute_total_mass(&mut self) {
        let mut tm = 0.;
        for i in 0..self.x.len() {
            tm += self.m[i];
        }

        self.total_mass = tm;
        assert!(self.total_mass > 0.);
    }

    /// compute center of mass
    pub fn compute_center_of_mass(&mut self) {
        let mut cm = Vector3::zero();
        for i in 0..self.x.len() {
            cm[0] += self.m[i] * self.x[i];
            cm[1] += self.m[i] * self.y[i];
            cm[2] += self.m[i] * self.z[i];
        }
        self.cm = cm / self.total_mass;
    }
    /// compute Moment of inertia tensor
    /// Refer http://www.kwon3d.com/theory/moi/iten.html
    pub fn compute_moment_of_inertia(&mut self) {
        // set the orientation of the body
        self.orientation = Matrix3::new(1., 0., 0., 0., 1., 0., 0., 0., 1.);
        let x = &self.x;
        let y = &self.y;
        let z = &self.z;
        let m = &self.m;
        let mut moi = Matrix3::zero();
        let mut i_xx = 0.;
        let mut i_yy = 0.;
        let mut i_zz = 0.;
        let mut i_xy = 0.;
        let mut i_xz = 0.;
        let mut i_yz = 0.;
        let i_yx;
        let i_zx;
        let i_zy;
        let mut xi;
        let mut yi;
        let mut zi;
        let mut mi;

        for i in 0..self.x.len() {
            xi = x[i] - self.cm[0];
            yi = y[i] - self.cm[1];
            zi = z[i] - self.cm[2];
            mi = m[i];
            i_xx += mi * (yi.powf(2.) + zi.powf(2.));
            i_yy += mi * (xi.powf(2.) + zi.powf(2.));
            i_zz += mi * (xi.powf(2.) + yi.powf(2.));
            i_xy += -mi * xi * yi;
            i_xz += -mi * xi * zi;
            i_yz += -mi * yi * zi;
        }
        // set the symmetric terms
        i_yx = i_xy;
        i_zx = i_xz;
        i_zy = i_yz;

        // set the moment of inertia. The matrix3 is in column major order
        // set the first column
        moi[0] = Vector3::new(i_xx, i_yx, i_zx);
        moi[1] = Vector3::new(i_xy, i_yy, i_zy);
        moi[2] = Vector3::new(i_xz, i_yz, i_zz);

        // find the inverse
        self.moi_body_inv = moi.invert().unwrap();
        self.moi_global_inv = self.orientation * moi.invert().unwrap();
    }

    pub fn save_initial_position_vectors(&mut self) {
        for i in 0..self.x.len() {
            self.r_body[i] = Vector3::new(
                self.x[i] - self.cm[0],
                self.y[i] - self.cm[1],
                self.z[i] - self.cm[2],
            );
        }
        self.r_dash = self.r_body.clone();
        self.r_dash0 = self.r_body.clone();
    }

    pub fn update_ang_vel(&mut self) {
        self.ang_vel = self.moi_global_inv * self.ang_mom;
    }

    pub fn initialize(&mut self) {
        // compute total mass
        self.compute_total_mass();
        // compute center of mass
        self.compute_center_of_mass();
        // compute center of mass
        self.compute_moment_of_inertia();
        // save the body coordinate system position vectors
        self.save_initial_position_vectors();
    }
}

pub fn apply_gravity(
    d_m: &[f32], d_fx: &mut [f32], d_fy: &mut [f32], d_fz: &mut [f32],
    gx: f32, gy: f32, gz: f32,

) {
    for i in 0..d_fx.len(){
        d_fx[i] = gx * d_m[i];
        d_fy[i] = gy * d_m[i];
        d_fz[i] = gz * d_m[i];
    }
}

pub fn linear_interparticle_force(
    d_x: &[f32], d_y: &[f32], d_z: &[f32], d_u: &[f32],
    d_v: &[f32], d_w: &[f32], d_rad: &[f32], d_fx: &mut [f32],
    d_fy: &mut [f32], d_fz: &mut [f32],

    s_x: &[f32], s_y: &[f32], s_z: &[f32], s_u: &[f32],
    s_v: &[f32], s_w: &[f32], s_rad: &[f32], s_nnps_id: usize,

    nnps: &NNPS,
    k_n: f32,
    eta_n: f32,
) {
    d_fx.par_iter_mut()
        .zip(d_fy.par_iter_mut().zip(d_fz.par_iter_mut().enumerate()))
        .for_each(
            |(d_fx_i, (d_fy_i, (i, d_fz_i)))| {
                let mut xij = Vector3::new(0., 0., 0.);
                let mut nij;
                let mut vij: Vector3<f32> = Vector3::new(0., 0., 0.);
                let mut vij_n: Vector3<f32>;
                let mut fij_n = Vector3::new(0., 0., 0.);

                let mut rij;
                let mut overlap_n;
                let nbrs = match nnps.dim {
                    1 => get_neighbours_1d(d_x[i], d_y[i], d_z[i], s_nnps_id, &nnps),
                    2 => get_neighbours_2d(d_x[i], d_y[i], d_z[i], s_nnps_id, &nnps),
                    3 => get_neighbours_3d(d_x[i], d_y[i], d_z[i], s_nnps_id, &nnps),
                    _ => panic!("Dimensions are wrong"),
                };

                for &j in nbrs.iter() {
                    // Reset the forces for next contact
                    fij_n[0] = 0.;
                    fij_n[1] = 0.;
                    fij_n[2] = 0.;

                    xij[0] = s_x[j] - d_x[i];
                    xij[1] = s_y[j] - d_y[i];
                    xij[2] = s_z[j] - d_z[i];
                    rij = xij.magnitude();

                    // eliminate self interaction
                    if rij > 1e-12 {
                        // overlap amount
                        overlap_n = d_rad[i] + s_rad[j] - rij;
                        if overlap_n > 0. {
                            // normal vector from i to j
                            // Be careful about this
                            nij = xij / rij;

                            // Relative velocity particle i w.r.t j is uij
                            vij[0] = d_u[i] - s_u[j];
                            vij[1] = d_v[i] - s_v[j];
                            vij[2] = d_w[i] - s_w[j];

                            // normal velocity is
                            vij_n = vij.dot(nij) * nij;

                            // --------------------------------
                            // normal force due to the normal overlap
                            fij_n = -k_n * overlap_n * nij - eta_n * vij_n;

                            *d_fx_i += fij_n[0];
                            *d_fy_i += fij_n[1];
                            *d_fz_i += fij_n[2];
                        }
                    }
                }
            });
}

fn normalize_matrix3(mat: &mut Matrix3<f32>){
    // refer https://cg.informatik.uni-freiburg.de/course_notes/sim_06_rigidBodies.pdf
    let b1 = mat.x / mat.x.magnitude();
    let b2 = mat.y - b1.dot(mat.y) * b1;
    let b2 = b2 / b2.magnitude();
    let b3 = mat.z - b1.dot(mat.z) * b1 - b2.dot(mat.z) * b2;
    let b3 = b3 / b3.magnitude();
    // setup the normalized matrix
    mat.x = b1;
    mat.y = b2;
    mat.z = b3;
}

impl EulerIntegrator for RB3d {
    fn euler_stage_1(&mut self, dt: f32) {
        // aggregate all the forces to act at center of mass, and similar
        // way compute the torque
        let mut f = Vector3::zero();
        let mut fi;
        let mut trq = Vector3::zero();
        for i in 0..self.x.len() {
            fi = Vector3::new(self.fx[i], self.fy[i], self.fz[i]);
            f += fi;
            trq += self.r_dash[i].cross(fi);
        }
        // set the total force and torque
        self.net_force = f;
        self.torque = trq;
        // evolve the center of mass and center of mass velocity to next time step (t + dt)
        self.lin_vel += f * dt;
        self.cm += self.lin_vel * dt;
        // Evolve orientation to next time step (t + dt)
        self.orientation +=
            dt * Matrix3::new(
                // first column
                0.,
                self.ang_vel[2],
                -self.ang_vel[1],
                // second column
                -self.ang_vel[2],
                0.,
                self.ang_vel[0],
                // third column
                self.ang_vel[1],
                -self.ang_vel[0],
                0.,
            ) * self.orientation;
        // normalize the orientation matrix
        normalize_matrix3(&mut self.orientation);

        // update angular momentum
        self.ang_mom += dt * trq;

        // compute the moment of inertia at current time by using
        // angular momentum and rotation matrix at time t+dt
        self.moi_global_inv = self.orientation * self.moi_body_inv * self.orientation.transpose();

        // Update the angular velocity from the angular momentum at time
        // t + dt and the moi tensor at time t + dt
        self.ang_vel = self.moi_global_inv * self.ang_mom;

        // Update the position vectors from center of mass to the partices
        // r_dash
        let r_dash = &mut self.r_dash;
        let r_body = &self.r_body;
        for i in 0..self.x.len() {
            r_dash[i] = self.orientation * r_body[i];
            self.x[i] = self.cm[0] + r_dash[i][0];
            self.y[i] = self.cm[1] + r_dash[i][1];
            self.z[i] = self.cm[2] + r_dash[i][2];
            // velocity due to angular effect
            let tmp = self.ang_vel.cross(r_dash[i]);
            self.u[i] = self.lin_vel[0] + tmp[0];
            self.v[i] = self.lin_vel[1] + tmp[1];
            self.w[i] = self.lin_vel[2] + tmp[2];
        }
    }
}

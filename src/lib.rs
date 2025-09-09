use std::f64::consts::PI;
use rayon::prelude::*;


#[derive(Debug)]
pub struct ParticlesDEM {
    pub name: String,

    pub x: Vec<f64>,
    pub y: Vec<f64>,
    pub z: Vec<f64>,
    pub u: Vec<f64>,
    pub v: Vec<f64>,
    pub w: Vec<f64>,
    pub wx: Vec<f64>,
    pub wy: Vec<f64>,
    pub wz: Vec<f64>,
    pub fx: Vec<f64>,
    pub fy: Vec<f64>,
    pub fz: Vec<f64>,
    pub torx: Vec<f64>,
    pub tory: Vec<f64>,
    pub torz: Vec<f64>,

    // Each particle influence radius
    pub h: Vec<f64>,

    pub m: Vec<f64>,
    pub moi: Vec<f64>,
    pub rho: Vec<f64>,
    pub radius: Vec<f64>,
    pub youngs_mod: Vec<f64>,
    pub poisson_ratio: Vec<f64>,
    pub shear_mod: Vec<f64>,

    // Properties to track the contacts
    pub dem_id: Vec<usize>,
    pub total_no_dem_tng_ss_contacts: Vec<usize>,
    pub dem_tng_disp_ss_idx: Vec<Vec<usize>>,
    pub dem_tng_disp_ss_x: Vec<Vec<f64>>,
    pub dem_tng_disp_ss_y: Vec<Vec<f64>>,
    pub dem_tng_disp_ss_z: Vec<Vec<f64>>,
    pub dem_tng_ss_dem_id: Vec<Vec<usize>>,

    pub dem_tng_disp_sw_x: Vec<Vec<f64>>,
    pub dem_tng_disp_sw_y: Vec<Vec<f64>>,
    pub dem_tng_disp_sw_z: Vec<Vec<f64>>,

    // Properties related to the nearest neighbour search
    pub neighbours: Neighbours,
}

impl ParticlesDEM {
    #[allow(non_snake_case)]
    pub fn new(dim: i64, name: String, x: &[f64], y: &[f64], z: &[f64], h: &[f64], rho: &[f64], radius: &[f64], dem_id: &[usize]) -> Self{
        let n = x.len();
        assert_eq!(y.len(), n);
        assert_eq!(z.len(), n);
        assert_eq!(rho.len(), n);
        assert_eq!(radius.len(), n);

        let mut m = Vec::with_capacity(n);
        let mut moi = Vec::with_capacity(n);

        for i in 0..n {
            let r = radius[i];
            let d = rho[i];
            let mut m_i = (4.0 / 3.0) * std::f64::consts::PI * r.powf(dim as f64) * d;
            let mut I_i = (2.0 / 5.0) * m_i * r.powf(2.);

            if dim == 2 {
                m_i = PI * r.powf(dim as f64) * d;
                I_i = 1.0 / 2.0 * m_i * r.powf(2.);
            }

            m.push(m_i);
            moi.push(I_i);
        }

        Self {
            name,
            x: x.to_vec(),
            y: y.to_vec(),
            z: z.to_vec(),

            u: vec![0.0; n],
            v: vec![0.0; n],
            w: vec![0.0; n],

            wx: vec![0.0; n],
            wy: vec![0.0; n],
            wz: vec![0.0; n],

            fx: vec![0.0; n],
            fy: vec![0.0; n],
            fz: vec![0.0; n],

            torx: vec![0.0; n],
            tory: vec![0.0; n],
            torz: vec![0.0; n],

            h: h.to_vec(),

            m: m,
            moi: moi,
            rho: rho.to_vec(),
            radius: radius.to_vec(),
            youngs_mod: vec![1e6; n],
            poisson_ratio: vec![0.23; n],
            shear_mod: vec![1e4; n],

            dem_id: dem_id.to_vec(),

            total_no_dem_tng_ss_contacts: vec![0; n],
            dem_tng_disp_ss_idx: vec![Vec::new(); n],
            dem_tng_ss_dem_id: vec![Vec::new(); n],
            dem_tng_disp_ss_x: vec![Vec::new(); n],
            dem_tng_disp_ss_y: vec![Vec::new(); n],
            dem_tng_disp_ss_z: vec![Vec::new(); n],

            dem_tng_disp_sw_x: vec![Vec::new(); n],
            dem_tng_disp_sw_y: vec![Vec::new(); n],
            dem_tng_disp_sw_z: vec![Vec::new(); n],

            neighbours: Neighbours,
        }
    }

}

macro_rules! generate_access_traits {
    ($struct_name:ident, { $($field:ident),* $(,)? }) => {
        $(
            paste::paste! {
                pub trait [<Has $field:camel>] {
                    fn [<get_ $field>](&self) -> &Vec<f64>;
                    fn [<get_ $field _mut>](&mut self) -> &mut Vec<f64>;
                }

                impl [<Has $field:camel>] for $struct_name {
                    fn [<get_ $field>](&self) -> &Vec<f64> {
                        &self.$field
                    }
                    fn [<get_ $field _mut>](&mut self) -> &mut Vec<f64> {
                        &mut self.$field
                    }
                }
            }
        )*
    };
}

generate_access_traits!(ParticlesDEM, {
    m, rho, moi, radius,
    youngs_mod, poisson_ratio, shear_mod,
    x, y, z, u, v, w, wx, wy, wz,
    fx, fy, fz, torx, tory, torz
});


// Output trait for every particle array
pub trait WriteOutput {
    fn write_vtk(&self, output: String, t: f64);
    fn write_hdf5(&self, output: String, t: f64);
}


impl WriteOutput for ParticlesDEM {
    fn write_vtk(&self, output: String, t: f64) {}
    fn write_hdf5(&self, output: String, t: f64) {}
}


pub trait GTVFStep {
    fn stage1(&mut self, dt: f64);
    fn stage2(&mut self, dt: f64);
    fn stage3(&mut self, dt: f64);
}


impl GTVFStep for ParticlesDEM {
    fn stage1(&mut self, dt: f64) {
        let dtb2 = 0.5 * dt;
        let n = self.u.len();

        for d_idx in 0..n {
            let m_inverse = 1. / self.m[d_idx];
            self.u[d_idx] += dtb2 * self.fx[d_idx] * m_inverse;
            self.v[d_idx] += dtb2 * self.fy[d_idx] * m_inverse;
            self.w[d_idx] += dtb2 * self.fz[d_idx] * m_inverse;

            let I_inverse = 1. / self.moi[d_idx];
            self.wx[d_idx] += dtb2 * self.torx[d_idx] * I_inverse;
            self.wy[d_idx] += dtb2 * self.tory[d_idx] * I_inverse;
            self.wz[d_idx] += dtb2 * self.torz[d_idx] * I_inverse;
        }
    }

    fn stage2(&mut self, dt: f64) {
        let n = self.u.len();
        for d_idx in 0..n {
            self.x[d_idx] += dt * self.u[d_idx];
            self.y[d_idx] += dt * self.v[d_idx];
            self.z[d_idx] += dt * self.w[d_idx];
        }
    }

    fn stage3(&mut self, dt: f64) {
        let dtb2 = 0.5 * dt;
        let n = self.u.len();

        for d_idx in 0..n {
            let m_inverse = 1. / self.m[d_idx];
            self.u[d_idx] += dtb2 * self.fx[d_idx] * m_inverse;
            self.v[d_idx] += dtb2 * self.fy[d_idx] * m_inverse;
            self.w[d_idx] += dtb2 * self.fz[d_idx] * m_inverse;

            let I_inverse = 1. / self.moi[d_idx];
            self.wx[d_idx] += dtb2 * self.torx[d_idx] * I_inverse;
            self.wy[d_idx] += dtb2 * self.tory[d_idx] * I_inverse;
            self.wz[d_idx] += dtb2 * self.torz[d_idx] * I_inverse;
        }
    }
}


/// Continuity equation to compute rate of change of density of a particle.
//
pub fn apply_hertz_contact_force_ss_DEM(
    d_x: &Vec<f64>, d_y: &Vec<f64>, d_z: &Vec<f64>,
    d_u: &Vec<f64>, d_v: &Vec<f64>, d_w: &Vec<f64>,
    d_wx: &Vec<f64>, d_wy: &Vec<f64>, d_wz: &Vec<f64>,
    d_fx: &mut Vec<f64>, d_fy: &mut Vec<f64>, d_fz: &mut Vec<f64>,
    d_torx: &mut Vec<f64>, d_tory: &mut Vec<f64>, d_torz: &mut Vec<f64>,

    d_h: &Vec<f64>,

    d_m: &Vec<f64>, d_moi: &Vec<f64>, d_rho: &Vec<f64>,
    d_radius: &Vec<f64>,
    d_youngs_mod: &Vec<f64>,
    d_poisson_ratio: &Vec<f64>,
    d_shear_mod: &Vec<f64>,

    d_dem_id: &Vec<usize>,

    d_total_no_dem_tng_ss_contacts: &mut Vec<usize>,
    d_dem_tng_disp_ss_idx: &mut Vec<Vec<usize>>,
    d_dem_tng_ss_dem_id: &mut Vec<Vec<usize>>,
    d_dem_tng_disp_ss_x: &mut Vec<Vec<f64>>,
    d_dem_tng_disp_ss_y: &mut Vec<Vec<f64>>,
    d_dem_tng_disp_ss_z: &mut Vec<Vec<f64>>,

    s_x: &Vec<f64>, s_y: &Vec<f64>, s_z: &Vec<f64>,
    s_u: &Vec<f64>, s_v: &Vec<f64>, s_w: &Vec<f64>,
    s_wx: &Vec<f64>, s_wy: &Vec<f64>, s_wz: &Vec<f64>,
    s_h: &Vec<f64>,
    s_m: &Vec<f64>, s_rho: &Vec<f64>,
    s_radius: &Vec<f64>,
    s_youngs_mod: &Vec<f64>,
    s_poisson_ratio: &Vec<f64>,
    s_shear_mod: &Vec<f64>,
    s_dem_id: &Vec<usize>,
    s_neighbours: &Neighbours,

    // time and time step
    t: f64, dt: f64,
    // constants like coefficient of restitution and friction coefficient
    cor_pp: f64, friction_pp: f64,
)
{
    d_fx
        .par_iter_mut()
        .zip(d_fy.par_iter_mut())
        .zip(d_fz.par_iter_mut())
        .zip(d_torx.par_iter_mut())
        .zip(d_tory.par_iter_mut())
        .zip(d_torz.par_iter_mut())

        .zip(d_total_no_dem_tng_ss_contacts.par_iter_mut())
        .zip(d_dem_tng_disp_ss_idx.par_iter_mut())
        .zip(d_dem_tng_ss_dem_id.par_iter_mut())
        .zip(d_dem_tng_disp_ss_x.par_iter_mut())
        .zip(d_dem_tng_disp_ss_y.par_iter_mut())
        .zip(d_dem_tng_disp_ss_z.par_iter_mut())
        .enumerate()
        .for_each(
            |(
                d_idx,
                (((((((((((d_fx_i, d_fy_i), d_fz_i,), d_torx_i,), d_tory_i,), d_torz_i,),
                       d_total_no_dem_tng_ss_contacts_i,), d_dem_tng_disp_ss_idx_i,),
                     d_dem_tng_ss_dem_id_i,),
                    d_dem_tng_disp_ss_x_i,), d_dem_tng_disp_ss_y_i,),
                  d_dem_tng_disp_ss_z_i,))| {

                // We will replace this line with s_idx in neighbours
                // Also compare the dem_id of this particle array
                let neighbours = s_neighbours.get_neighbours(radius);
                for s_idx in neighbours{
                    if d_dem_id[d_idx] != s_dem_id[s_idx] || d_idx != s_idx {
                        // ========================================
                        // Initialize the common primitives
                        // ========================================
                        // Positions
                        let pos_i = [d_x[d_idx], d_y[d_idx], d_z[d_idx]];
                        let pos_j = [s_x[s_idx], s_y[s_idx], s_z[s_idx]];

                        // Vector from j to i
                        let pos_ij = [
                            pos_i[0] - pos_j[0],
                            pos_i[1] - pos_j[1],
                            pos_i[2] - pos_j[2],
                        ];

                        // Squared distance
                        let r2ij = pos_ij[0].powi(2) + pos_ij[1].powi(2) + pos_ij[2].powi(2);
                        // Distance
                        let rij = r2ij.sqrt();
                        // ========================================
                        // End fo initialize the common primitives
                        // ========================================

                        // Overlap
                        let overlap = d_radius[d_idx] + s_radius[s_idx] - rij;

                        let a_i = d_radius[d_idx] - overlap / 2.;
                        let a_j = s_radius[s_idx] - overlap / 2.;

                        // Normal vector from j to i
                        let nij_x = pos_ij[0] / rij;
                        let nij_y = pos_ij[1] / rij;
                        let nij_z = pos_ij[2] / rij;

                        // Velocity of i at contact
                        let mut vel_i = [0.0; 3];
                        vel_i[0] = d_u[d_idx] + (d_wy[d_idx] * nij_z - d_wz[d_idx] * nij_y) * a_i;
                        vel_i[1] = d_v[d_idx] + (d_wz[d_idx] * nij_x - d_wx[d_idx] * nij_z) * a_i;
                        vel_i[2] = d_w[d_idx] + (d_wx[d_idx] * nij_y - d_wy[d_idx] * nij_x) * a_i;

                        // Velocity of j at contact
                        let mut vel_j = [0.0; 3];
                        vel_j[0] = s_u[s_idx] + (-s_wy[s_idx] * nij_z + s_wz[s_idx] * nij_y) * a_j;
                        vel_j[1] = s_v[s_idx] + (-s_wz[s_idx] * nij_x + s_wx[s_idx] * nij_z) * a_j;
                        vel_j[2] = s_w[s_idx] + (-s_wx[s_idx] * nij_y + s_wy[s_idx] * nij_x) * a_j;

                        // Relative velocity at the contact point
                        let vel_ij = [
                            vel_i[0] - vel_j[0],
                            vel_i[1] - vel_j[1],
                            vel_i[2] - vel_j[2],
                        ];

                        // Normal velocity magnitude
                        let vij_dot_nij = vel_ij[0] * nij_x + vel_ij[1] * nij_y + vel_ij[2] * nij_z;
                        let vn_x = vij_dot_nij * nij_x;
                        let vn_y = vij_dot_nij * nij_y;
                        let vn_z = vij_dot_nij * nij_z;

                        // Tangential velocity
                        let vt_x = vel_ij[0] - vn_x;
                        let vt_y = vel_ij[1] - vn_y;
                        let vt_z = vel_ij[2] - vn_z;

                        // find the force if the particles are overlapping
                        if (overlap > 0.) {
                            // Check if the particle is being tracked
                            // if the particle is being tracked then update the
                            // tracked tangential index
                            let mut found = 0;
                            // Remember this is a sentinal values. We will just initialize
                            // it to some value which will be overridden below
                            let mut found_at = usize::MAX;

                            for k in 0..*d_total_no_dem_tng_ss_contacts_i {
                                if s_idx == d_dem_tng_disp_ss_idx_i[k] && s_dem_id[s_idx] == d_dem_tng_ss_dem_id_i[k] {
                                    found = 1;
                                    found_at = k;
                                    break;
                                }
                            }

                            if found == 1 {
                                // don't do anything
                            } else {
                                found = 1;
                                // implies this is a new contact
                                // so add it to the contacting indices and increase the total
                                // count of the contacting indices

                                // create a new index at the end of the tracking
                                *d_total_no_dem_tng_ss_contacts_i += 1;
                                found_at = (*d_total_no_dem_tng_ss_contacts_i - 1);
                                d_dem_tng_disp_ss_idx_i.push(s_idx);
                                d_dem_tng_ss_dem_id_i.push(s_dem_id[s_idx]);
                                d_dem_tng_disp_ss_x_i.push(0.0);
                                d_dem_tng_disp_ss_y_i.push(0.0);
                                d_dem_tng_disp_ss_z_i.push(0.0);
                            }
                            // Compute stiffness

                            // effective Young's modulus
                            let mut tmp_1 = (1.0 - (d_poisson_ratio[d_idx] * d_poisson_ratio[d_idx])) / d_youngs_mod[d_idx];
                            let mut tmp_2 = (1.0 - (s_poisson_ratio[s_idx] * s_poisson_ratio[s_idx])) / s_youngs_mod[s_idx];
                            let E_eff = 1.0 / (tmp_1 + tmp_2);

                            let tmp_3 = 1.0 / d_radius[d_idx];
                            let tmp_4 = 1.0 / s_radius[s_idx];
                            let R_eff = 1.0 / (tmp_3 + tmp_4);

                            // Eq 4 [1]
                            let kn = 4.0 / 3.0 * E_eff * R_eff.powf(0.5);

                            // compute damping coefficient
                            tmp_1 = cor_pp.ln();
                            tmp_2 = tmp_1 * tmp_1 + std::f64::consts::PI * std::f64::consts::PI;
                            let beta = tmp_1 / tmp_2.powf(0.5);
                            let S_n = 2.0 * E_eff * (R_eff * overlap).powf(0.5);
                            tmp_1 = 1.0 / d_m[d_idx];
                            tmp_2 = 1.0 / s_m[s_idx];
                            let m_eff = 1.0 / (tmp_1 + tmp_2);
                            let eta_n = -2.0 * (5.0 / 6.0f64).powf(0.5) * beta * (S_n * m_eff).powf(0.5);

                            // Compute the tangential stiffness
                            tmp_1 = (2.0 - d_poisson_ratio[d_idx]) / d_shear_mod[d_idx];
                            tmp_2 = (2.0 - s_poisson_ratio[s_idx]) / s_shear_mod[s_idx];
                            let G_eff = 1.0 / (tmp_1 + tmp_2);

                            // Eq 12 [1]
                            let kt = 8.0 * G_eff * (R_eff * overlap).powf(0.5);
                            let S_t = kt;
                            let eta_t = -2.0 * (5.0 / 6.0f64).powf(0.5) * beta * (S_t * m_eff).powf(0.5);

                            // normal force
                            let fn_ = kn * overlap.powf(1.5);
                            let fn_x = fn_ * nij_x - eta_n * vn_x;
                            let fn_y = fn_ * nij_y - eta_n * vn_y;
                            let fn_z = fn_ * nij_z - eta_n * vn_z;

                            // Tangential force
                            let mut tangential_frc_ss_x_i = 0.;
                            let mut tangential_frc_ss_y_i = 0.;
                            let mut tangential_frc_ss_z_i = 0.;
                            // Incremenet the tangential force and add it to the total force
                            // check if there is relative motion
                            let vij_magn = (vel_ij[0]*vel_ij[0] + vel_ij[1]*vel_ij[1] + vel_ij[2]*vel_ij[2]).powf(0.5);
                            if (vij_magn < 1e-12) {
                                // make the tangential displacement
                                d_dem_tng_disp_ss_x_i[found_at] = 0.;
                                d_dem_tng_disp_ss_y_i[found_at] = 0.;
                                d_dem_tng_disp_ss_z_i[found_at] = 0.;
                            }
                            else{
                                // the tangential vector is
                                let tx_tmp = vel_ij[0] - nij_x * vij_dot_nij;
                                let ty_tmp = vel_ij[1] - nij_y * vij_dot_nij;
                                let tz_tmp = vel_ij[2] - nij_z * vij_dot_nij;

                                let ti_magn = (tx_tmp*tx_tmp + ty_tmp*ty_tmp + tz_tmp*tz_tmp).powf(0.5);

                                let mut ti_x = 0.;
                                let mut ti_y = 0.;
                                let mut ti_z = 0.;

                                if (ti_magn > 1e-12){
                                    ti_x = tx_tmp / ti_magn;
                                    ti_y = ty_tmp / ti_magn;
                                    ti_z = tz_tmp / ti_magn;
                                }

                                // this is correct
                                let delta_lt_x_star = d_dem_tng_disp_ss_x_i[found_at] + vel_ij[0] * dt;
                                let delta_lt_y_star = d_dem_tng_disp_ss_y_i[found_at] + vel_ij[1] * dt;
                                let delta_lt_z_star = d_dem_tng_disp_ss_z_i[found_at] + vel_ij[2] * dt;

                                let delta_lt_dot_ti = (delta_lt_x_star * ti_x +
                                                       delta_lt_y_star * ti_y +
                                                       delta_lt_z_star * ti_z);

                                d_dem_tng_disp_ss_x_i[found_at] = delta_lt_dot_ti * ti_x;
                                d_dem_tng_disp_ss_y_i[found_at] = delta_lt_dot_ti * ti_y;
                                d_dem_tng_disp_ss_z_i[found_at] = delta_lt_dot_ti * ti_z;

                                let ft_x_star = -kt * d_dem_tng_disp_ss_x_i[found_at] - eta_t * vt_x;
                                let ft_y_star = -kt * d_dem_tng_disp_ss_y_i[found_at] - eta_t * vt_y;
                                let ft_z_star = -kt * d_dem_tng_disp_ss_z_i[found_at] - eta_t * vt_z;

                                let ft_magn = (ft_x_star*ft_x_star + ft_y_star*ft_y_star + ft_z_star*ft_z_star).powf(0.5);
                                let fn_magn = (fn_x*fn_x + fn_y*fn_y + fn_z*fn_z).powf(0.5);

                                let mut ft_magn_star = ft_magn;
                                if (ft_magn_star > friction_pp * fn_magn){
                                    ft_magn_star = friction_pp * fn_magn;
                                }

                                // compute the tangential force, by equation 27
                                tangential_frc_ss_x_i = -ft_magn_star * ti_x;
                                tangential_frc_ss_y_i = -ft_magn_star * ti_y;
                                tangential_frc_ss_z_i = -ft_magn_star * ti_z;

                                let modified_delta_lt_x = -tangential_frc_ss_x_i / kt;
                                let modified_delta_lt_y = -tangential_frc_ss_y_i / kt;
                                let modified_delta_lt_z = -tangential_frc_ss_z_i / kt;

                                let lt_magn = (modified_delta_lt_x*modified_delta_lt_x + modified_delta_lt_y*modified_delta_lt_y +
                                               modified_delta_lt_z*modified_delta_lt_z).powf(0.5);

                                if (lt_magn > 1e-12){
                                    d_dem_tng_disp_ss_x_i[found_at] = modified_delta_lt_x / lt_magn;
                                    d_dem_tng_disp_ss_y_i[found_at] = modified_delta_lt_y / lt_magn;
                                    d_dem_tng_disp_ss_z_i[found_at] = modified_delta_lt_z / lt_magn;
                                }
                                else {
                                    d_dem_tng_disp_ss_x_i[found_at] = 0.;
                                    d_dem_tng_disp_ss_y_i[found_at] = 0.;
                                    d_dem_tng_disp_ss_z_i[found_at] = 0.;
                                }
                            }

                            // Add force to the particle i due to contact with particle j
                            *d_fx_i += fn_x + tangential_frc_ss_x_i;
                            *d_fy_i += fn_y + tangential_frc_ss_y_i;
                            *d_fz_i += fn_z + tangential_frc_ss_z_i;
                        }
                    }
                }
            });
}

macro_rules! apply_hertz_contact_force_ss_DEM_macro {
    ($dest:ident, ($($sources:ident),*), $t:expr, $dt:expr, $cor_pp:expr, $friction_pp:expr) => {
        $(
            apply_hertz_contact_force_ss_DEM(
                // Destination (DEM) particle data
                &$dest.x, &$dest.y, &$dest.z,
                &$dest.u, &$dest.v, &$dest.w,
                &$dest.wx, &$dest.wy, &$dest.wz,
                &mut $dest.fx, &mut $dest.fy, &mut $dest.fz,
                &mut $dest.torx, &mut $dest.tory, &mut $dest.torz,

                &$dest.h,

                &$dest.m, &$dest.moi, &$dest.rho,
                &$dest.radius,
                &$dest.youngs_mod,
                &$dest.poisson_ratio,
                &$dest.shear_mod,

                &$dest.dem_id,

                &mut $dest.total_no_dem_tng_ss_contacts,
                &mut $dest.dem_tng_disp_ss_idx,
                &mut $dest.dem_tng_ss_dem_id,
                &mut $dest.dem_tng_disp_ss_x,
                &mut $dest.dem_tng_disp_ss_y,
                &mut $dest.dem_tng_disp_ss_z,

                // Source (neighbor) particle data
                &$sources.x, &$sources.y, &$sources.z,
                &$sources.u, &$sources.v, &$sources.w,
                &$sources.wx, &$sources.wy, &$sources.wz,
                &$sources.neighbour_radius,
                &$sources.m, &$sources.rho,
                &$sources.radius,
                &$sources.youngs_mod,
                &$sources.poisson_ratio,
                &$sources.shear_mod,
                &$sources.dem_id,
                &$sources.neighbours,

                // Time
                $t, $dt,

                // Contact properties
                $cor_pp, $friction_pp
            );
        )*
    };
}


pub fn update_dem_tng_ss_contacts(
    d_x: &Vec<f64>, d_y: &Vec<f64>, d_z: &Vec<f64>,
    d_m: &Vec<f64>,
    d_radius: &Vec<f64>,
    d_dem_id: &Vec<usize>,

    d_total_no_dem_tng_ss_contacts: &mut Vec<usize>,
    d_dem_tng_disp_ss_idx: &mut Vec<Vec<usize>>,
    d_dem_tng_ss_dem_id: &mut Vec<Vec<usize>>,
    d_dem_tng_disp_ss_x: &mut Vec<Vec<f64>>,
    d_dem_tng_disp_ss_y: &mut Vec<Vec<f64>>,
    d_dem_tng_disp_ss_z: &mut Vec<Vec<f64>>,

    s_x: &Vec<f64>, s_y: &Vec<f64>, s_z: &Vec<f64>,
    s_m: &Vec<f64>,
    s_radius: &Vec<f64>,
    s_dem_id: &Vec<usize>,

    // time and time step
    t: f64, dt: f64,
)
{
    d_total_no_dem_tng_ss_contacts
        .par_iter_mut()
        .zip(d_dem_tng_disp_ss_idx.par_iter_mut())
        .zip(d_dem_tng_ss_dem_id.par_iter_mut())
        .zip(d_dem_tng_disp_ss_x.par_iter_mut())
        .zip(d_dem_tng_disp_ss_y.par_iter_mut())
        .zip(d_dem_tng_disp_ss_z.par_iter_mut())
        .enumerate()
        .for_each(
            |(d_idx,
              (((((
                  d_total_no_dem_tng_ss_contacts_i, d_dem_tng_disp_ss_idx_i,),
                  d_dem_tng_ss_dem_id_i), d_dem_tng_disp_ss_x_i,), d_dem_tng_disp_ss_y_i,),
               d_dem_tng_disp_ss_z_i,))| {
                let mut count = 0;
                let no_cnts_i = *d_total_no_dem_tng_ss_contacts_i;
                let mut k = 0;
                while count < no_cnts_i{
                    let s_idx = d_dem_tng_disp_ss_idx_i[k];
                    if d_dem_tng_ss_dem_id_i[k] == s_dem_id[s_idx] {
                        // Positions
                        let pos_i = [d_x[d_idx], d_y[d_idx], d_z[d_idx]];
                        let pos_j = [s_x[s_idx], s_y[s_idx], s_z[s_idx]];

                        // Vector from j to i
                        let pos_ij = [
                            pos_i[0] - pos_j[0],
                            pos_i[1] - pos_j[1],
                            pos_i[2] - pos_j[2],
                        ];

                        // Squared distance
                        let r2ij = pos_ij[0].powi(2) + pos_ij[1].powi(2) + pos_ij[2].powi(2);
                        // Distance
                        let rij = r2ij.sqrt();

                        // Overlap
                        let overlap = d_radius[d_idx] + s_radius[s_idx] - rij;

                        // This condition implies the particles are no more in contact
                        if (overlap < 0.) {
                            *d_total_no_dem_tng_ss_contacts_i -= 1;
                            d_dem_tng_disp_ss_idx_i.remove(k);
                            d_dem_tng_ss_dem_id_i.remove(k);
                            d_dem_tng_disp_ss_x_i.remove(k);
                            d_dem_tng_disp_ss_y_i.remove(k);
                            d_dem_tng_disp_ss_z_i.remove(k);
                        }
                        else {
                            k += 1;
                        }
                        count += 1;
                    }
                }
            });
}


#[macro_export]
macro_rules! update_dem_tng_ss_contacts_macro {
    ($dest:ident, ($($sources:ident),*), $t:expr, $dt:expr) => {
        $(
            update_dem_tng_pp_contacts(
                &$dest.x, &$dest.y, &$dest.z,
                &$dest.m,
                &$dest.radius,
                &$dest.dem_id,

                &mut $dest.total_no_dem_tng_ss_contacts,
                &mut $dest.dem_tng_disp_ss_idx,
                &mut $dest.dem_tng_ss_dem_id,
                &mut $dest.dem_tng_disp_ss_x,
                &mut $dest.dem_tng_disp_ss_y,
                &mut $dest.dem_tng_disp_ss_z,

                &$sources.x, &$sources.y, &$sources.z,
                &$sources.m,
                &$sources.radius,
                &$sources.dem_id,

                $t,
                $dt,
            );
        )*
    };
}

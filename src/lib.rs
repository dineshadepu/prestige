use std::f64::consts::PI;


#[derive(Debug)]
pub struct ParticlesDEM {
    pub name: String,
    pub m: Vec<f64>,
    pub rho: Vec<f64>,
    pub moi: Vec<f64>,
    pub radius: Vec<f64>,
    pub youngs_mod: Vec<f64>,
    pub poisson_ratio: Vec<f64>,
    pub shear_mod: Vec<f64>,
    pub x: Vec<f64>,
    pub y: Vec<f64>,
    pub z: Vec<f64>,
    pub u: Vec<f64>,
    pub v: Vec<f64>,
    pub w: Vec<f64>,
    pub fx: Vec<f64>,
    pub fy: Vec<f64>,
    pub fz: Vec<f64>,
}

impl ParticlesDEM {
    #[allow(non_snake_case)]
    pub fn new(dim: i64, name: String, x: &[f64], y: &[f64], z: &[f64], rho: &[f64], radius: &[f64]) -> Self{
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
            rho: rho.to_vec(),
            radius: radius.to_vec(),
            m: m,
            moi: moi,

            // Initialize other fields to default values
            youngs_mod: vec![1e6; n],
            poisson_ratio: vec![0.23; n],
            shear_mod: vec![1e4; n],

            u: vec![0.0; n],
            v: vec![0.0; n],
            w: vec![0.0; n],

            fx: vec![0.0; n],
            fy: vec![0.0; n],
            fz: vec![0.0; n],
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
    x, y, z, u, v, w,
    fx, fy, fz
});


/// Continuity equation to compute rate of change of density of a particle.
//
pub fn force_dem(
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

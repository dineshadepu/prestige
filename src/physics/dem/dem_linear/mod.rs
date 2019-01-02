pub mod equations;
use cgmath::Vector3;
use std::collections::HashMap;

/// Tangential Contact information
#[derive(Debug, Clone)]
pub struct TangCt {
    normal_0: Vector3<f32>,
    delta_tng: Vector3<f32>,
}

impl TangCt{
    pub fn new(normal_0: &Vector3<f32>, delta_tng: Vector3<f32>) -> TangCt{
        TangCt{
            normal_0: normal_0.clone(),
            delta_tng: delta_tng.clone(),
        }
    }
}

pub struct DEMLinear {
    pub x: Vec<f32>,
    pub y: Vec<f32>,
    pub z: Vec<f32>,
    pub x0: Vec<f32>,
    pub y0: Vec<f32>,
    pub z0: Vec<f32>,
    pub u: Vec<f32>,
    pub v: Vec<f32>,
    pub w: Vec<f32>,
    pub u0: Vec<f32>,
    pub v0: Vec<f32>,
    pub w0: Vec<f32>,
    pub omega_x: Vec<f32>,
    pub omega_y: Vec<f32>,
    pub omega_z: Vec<f32>,
    pub omega_x0: Vec<f32>,
    pub omega_y0: Vec<f32>,
    pub omega_z0: Vec<f32>,
    pub h: Vec<f32>,
    pub moi: Vec<f32>,
    pub rad: Vec<f32>,
    pub fx: Vec<f32>,
    pub fy: Vec<f32>,
    pub fz: Vec<f32>,
    pub tx: Vec<f32>,
    pub ty: Vec<f32>,
    pub tz: Vec<f32>,
    pub dem_idx: usize,
    pub total_dem_entities: usize,
    pub nnps_idx: usize,
    pub tng_ctcs: Vec<Vec<HashMap<usize, TangCt>>>,
}

impl DEMLinear {
    /// Creates a DEMLinear entity from x, y, z and radius of given particles
    pub fn from_xyzr(
        x: Vec<f32>,
        y: Vec<f32>,
        z: Vec<f32>,
        rad: Vec<f32>,
        dem_idx: usize,
        total_dem_entities: usize,
    ) -> DEMLinear {
        DEMLinear {
            x: x.clone(),
            y: y.clone(),
            z: z.clone(),
            rad: rad.clone(),
            x0: vec![0.; x.len()],
            y0: vec![0.; x.len()],
            z0: vec![0.; x.len()],
            u: vec![0.; x.len()],
            v: vec![0.; x.len()],
            w: vec![0.; x.len()],
            u0: vec![0.; x.len()],
            v0: vec![0.; x.len()],
            w0: vec![0.; x.len()],
            omega_x: vec![0.; x.len()],
            omega_y: vec![0.; x.len()],
            omega_z: vec![0.; x.len()],
            omega_x0: vec![0.; x.len()],
            omega_y0: vec![0.; x.len()],
            omega_z0: vec![0.; x.len()],
            h: vec![0.; x.len()],
            moi: vec![0.; x.len()],
            fx: vec![0.; x.len()],
            fy: vec![0.; x.len()],
            fz: vec![0.; x.len()],
            tx: vec![0.; x.len()],
            ty: vec![0.; x.len()],
            tz: vec![0.; x.len()],
            dem_idx: dem_idx,
            total_dem_entities: total_dem_entities,
            nnps_idx: 0,
            tng_ctcs: vec![vec![HashMap::new(); total_dem_entities]; x.len()],
        }
    }

    pub fn from_xyr(
        x: Vec<f32>,
        y: Vec<f32>,
        rad: Vec<f32>,
        dem_idx: usize,
        total_dem_entities: usize,
    ) -> DEMLinear {
        DEMLinear::from_xyzr(x, y, vec![0.; rad.len()], rad, dem_idx, total_dem_entities)
    }
}

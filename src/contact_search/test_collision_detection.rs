use crate::{
    contact_search::{GetXYZH},
    impl_GetXYZH,
};
use crate::geometry::grid_arange;
pub struct SimpleXYZ {
    pub x: Vec<f32>,
    pub y: Vec<f32>,
    pub z: Vec<f32>,
    pub h: Vec<f32>,
    pub nnps_idx: usize,
}

impl SimpleXYZ {
    pub fn new_with_xyz(x: Vec<f32>, y: Vec<f32>, z: Vec<f32>, nnps_idx: usize) -> SimpleXYZ {
        SimpleXYZ {
            x: x.clone(),
            y: y,
            z: z,
            h: vec![0.; x.len()],
            nnps_idx: nnps_idx,
        }
    }
    pub fn new_with_xy(x: Vec<f32>, y: Vec<f32>, nnps_idx: usize) -> SimpleXYZ {
        SimpleXYZ::new_with_xyz(x.clone(), y, vec![0.; x.len()], nnps_idx)
    }
}

impl_GetXYZH!(SimpleXYZ);


pub fn create_a_simple_2d_test_entity(spacing: f32) -> SimpleXYZ{
    let (x, y) = grid_arange(0.0, 1.0, spacing, 0.0, 1.0, spacing);
    let entity = SimpleXYZ::new_with_xy(x.clone(), y.clone(), 0);
    entity
}

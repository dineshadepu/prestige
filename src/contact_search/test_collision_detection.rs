use crate::geometry::grid_arange;
use crate::{
    contact_search::{GetXYZH, NNPSGeneric},
    impl_GetXYZH,
};

use rand::Rng;

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

pub fn create_a_simple_2d_test_entity(spacing: f32) -> SimpleXYZ {
    let (x, y) = grid_arange(0.0, 1.0, spacing, 0.0, 1.0, spacing);
    let entity = SimpleXYZ::new_with_xy(x.clone(), y.clone(), 0);
    entity
}

pub fn get_brute_force_nbrs(xi: f32, yi: f32, x: &[f32], y: &[f32], radius: f32) -> Vec<usize> {
    let mut brute_nbrs = vec![];
    for j in 0..x.len() {
        let rij = ((xi - x[j]).powf(2.) + (yi - y[j]).powf(2.)).sqrt();
        if rij < radius {
            brute_nbrs.push(j);
        }
    }
    brute_nbrs.sort();
    brute_nbrs
}

pub fn test_nbrs<T: GetXYZH, U: NNPSGeneric>(entities: Vec<&T>, nnps: &U, radius: f32) {
    let mut rng = rand::thread_rng();

    for ent1 in &entities {
        for ent2 in &entities {
            let (x1, y1, z1, _) = ent1.get_xyzh();
            let (x2, y2, z2, _) = ent2.get_xyzh();
            let nnps_id2 = ent2.get_nnps_id();

            // take 10 points from entity 1, and check neighbours of
            // those ten points against entity 2
            for _ in 0..10 {
                // we select random particles
                let i = rng.gen_range(0, x1.len());
                // we are asking for the neighbours of point at position
                // x[i], y[i] with an particles having an nnps_id2
                let nbrs = nnps.get_neighbours(x1[i], y1[i], z1[i], nnps_id2);
                let mut filtered_nbrs = vec![];
                // select the neighbours which are in limit or radius scale.
                for &j in nbrs.iter() {
                    let rij = ((x1[i] - x2[j]).powf(2.)
                               + (y1[i] - y2[j]).powf(2.)
                               + (z1[i] - z2[j]).powf(2.))
                        .sqrt();
                    if rij < radius {
                        filtered_nbrs.push(j);
                    }
                }
                // println!("{:?}", filtered_nbrs);
                // sort the filtered neighbours
                filtered_nbrs.sort();

                // compute neighbours by1 brute force
                let brute_nbrs = get_brute_force_nbrs(x1[i], y1[i], x2, y2, radius);
                assert_eq!(filtered_nbrs, brute_nbrs);
            }
        }
    }
}

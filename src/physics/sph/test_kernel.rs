use super::kernel::*;
use assert_approx_eq::assert_approx_eq;
use std::f32::consts::{FRAC_1_PI, PI};

fn sin_approximation(pos: &Vec<f32>, h: &Vec<f32>, kernel: &Kernel) -> Vec<f32> {
    let spacing = pos[2] - pos[1];
    let sin_actual = pos.iter().map(|val| val.sin()).collect::<Vec<f32>>();
    let mut sin_sph = vec![0.; pos.len()];
    for i in 0..pos.len() {
        for j in 0..pos.len() {
            // position between particles
            let rij = (pos[i] - pos[j]).abs();
            sin_sph[i] += sin_actual[j] * kernel.get_wij(rij, h[i]) * spacing;
        }
    }
    sin_sph
}
fn sin_deriv_approximation(pos: &Vec<f32>, h: &Vec<f32>, kernel: &Kernel) -> Vec<f32> {
    // create a sin array
    let spacing = pos[2] - pos[1];
    let sin_actual = pos.iter().map(|val| val.sin()).collect::<Vec<f32>>();
    let mut sin_deriv = vec![0.; pos.len()];
    for i in 0..pos.len() {
        let mut dwij = vec![0., 0., 0.];
        for j in 0..pos.len() {
            // position between particles
            let xij = vec![(pos[i] - pos[j]), 0., 0.];
            let rij = (pos[i] - pos[j]).abs();

            // compute gradient
            kernel.get_dwij(&xij, &mut dwij, rij, h[i]);

            sin_deriv[i] += sin_actual[j] * dwij[0] * spacing;
        }
    }
    sin_deriv
}

#[test]
fn test_cubic_kernel_attributes() {
    let ck = CubicKernel::new(2).unwrap();
    assert_eq!(ck.sigma, 10. / 7. * FRAC_1_PI);
    let ck = CubicKernel::new(1).unwrap();
    assert_eq!(ck.sigma, 2. / 3.);
}

#[test]
fn test_ck_wij() {
    let pos = crate::geometry::linspace(0., PI / 2., 100).collect::<Vec<_>>();
    let spacing = pos[2] - pos[1];
    let h = vec![1.2 * spacing; pos.len()];
    let ck = CubicKernel::new(1).unwrap();
    let sin_actual = pos.iter().map(|val| val.sin()).collect::<Vec<f32>>();
    let sin_computed = sin_approximation(&pos, &h, &ck);
    for i in 5..sin_actual.len() - 5 {
        assert_approx_eq!(sin_computed[i], sin_actual[i], 1e-2);
    }
}

#[test]
pub fn test_ck_dwij() {
    let pos = crate::geometry::linspace(0., PI / 2., 1000).collect::<Vec<_>>();
    let spacing = pos[2] - pos[1];
    let h = vec![1.2 * spacing; pos.len()];
    let ck = CubicKernel::new(1).unwrap();
    let sin_deriv_actual = pos.iter().map(|val| val.cos()).collect::<Vec<f32>>();
    let sin_deriv_sph = sin_deriv_approximation(&pos, &h, &ck);
    for i in 5..sin_deriv_actual.len() - 5 {
        assert_approx_eq!(sin_deriv_sph[i], sin_deriv_actual[i], 1e-1);
    }
}

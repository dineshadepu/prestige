use super::lucy_kernel::LucyKernel;
use super::test_kernel::{sin_approximation, sin_deriv_approximation};
use std::f32::consts::{FRAC_1_PI, PI};
use assert_approx_eq::assert_approx_eq;

#[test]
fn test_lucy_kernel_attributes() {
    let lk = LucyKernel::new(3).unwrap();
    assert_eq!(lk.sigma, 105. / 16. * FRAC_1_PI);
    let lk = LucyKernel::new(2).unwrap();
    assert_eq!(lk.sigma, 5. * FRAC_1_PI);
    let lk = LucyKernel::new(1).unwrap();
    assert_eq!(lk.sigma, 5. / 4.);
}

#[test] #[ignore]
fn test_lk_wij() {
    let pos = crate::geometry::linspace(0., PI / 2., 100).collect::<Vec<_>>();
    let spacing = pos[2] - pos[1];
    let h = vec![1.2 * spacing; pos.len()];
    let lk = LucyKernel::new(1).unwrap();
    let sin_actual = pos.iter().map(|val| val.sin()).collect::<Vec<f32>>();
    let sin_computed = sin_approximation(&pos, &h, &lk);
    for i in 5..sin_actual.len() - 5 {
        assert_approx_eq!(sin_computed[i], sin_actual[i], 1e-2);
    }
}

// #[test]
// pub fn test_ck_dwij() {
//     let pos = crate::geometry::linspace(0., PI / 2., 1000).collect::<Vec<_>>();
//     let spacing = pos[2] - pos[1];
//     let h = vec![1.2 * spacing; pos.len()];
//     let ck = LucyKernel::new(1).unwrap();
//     let sin_deriv_actual = pos.iter().map(|val| val.cos()).collect::<Vec<f32>>();
//     let sin_deriv_sph = sin_deriv_approximation(&pos, &h, &ck);
//     for i in 5..sin_deriv_actual.len() - 5 {
//         assert_approx_eq!(sin_deriv_sph[i], sin_deriv_actual[i], 1e-1);
//     }
// }

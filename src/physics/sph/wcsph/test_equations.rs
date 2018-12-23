#[cfg(test)]
use super::equations::*;

#[test]
fn test_rand() {
    use rand::Rng;
    // check for more
    // Hts://rust-lang-nursery.github.io/rust-cookbook/algorithms/randomness.html
    let mut rng = rand::thread_rng();
    assert!(rng.gen_range(0.0, 1000.0) > 0.);
}

#[test]
fn test_equation_of_state() {
    let num_pars = 10;
    let mut d_p = vec![0.; num_pars];
    let mut d_p_test = vec![0.; num_pars];

    // random number generator
    let mut rng = rand::thread_rng();
    let mut d_rho = (0..num_pars)
        .map(|_| rng.gen_range(0.0, 1000.0))
        .collect::<Vec<f32>>();
    let d_rho0 = 1000.;
    let c: f32 = 10.;
    let gamma = 7.;

    // compute pressure to check against
    let b = c.powf(2.) * d_rho0 / gamma;
    for i in 0..num_pars {
        d_p_test[i] = b * ((d_rho[i] / d_rho0).powf(7.) - 1.);
    }

    // now test the equation
    equation_of_state(&mut d_p, &d_rho, d_rho0, gamma, c);

    assert_eq!(d_p_test, d_p);
}

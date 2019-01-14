use super::Kernel;

pub fn sin_approximation(pos: &Vec<f32>, h: &Vec<f32>, kernel: &Kernel) -> Vec<f32> {
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
pub fn sin_deriv_approximation(pos: &Vec<f32>, h: &Vec<f32>, kernel: &Kernel) -> Vec<f32> {
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

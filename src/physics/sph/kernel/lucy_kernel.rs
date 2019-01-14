use super::Kernel;
use std::f32::consts::FRAC_1_PI;

/// Taken from Lucy 1977 original SPH paper
/// Equation 3.4 in G.R Liu SPH book
pub struct LucyKernel {
    /// dimension of the simulation
    pub dim: f32,
    /// scale factor should be divided by h.powf(self.dim)
    /// to get sigma_d
    pub sigma: f32,
}

impl LucyKernel {
    pub fn new(dim: usize) -> Result<LucyKernel, &'static str> {
        let tmp = match dim {
            1 => Some(5. / 4.),
            2 => Some(5. * FRAC_1_PI),
            3 => Some(105. / 16. * FRAC_1_PI),
            _ => None,
        };

        if let Some(sigma) = tmp {
            Ok(LucyKernel {
                dim: dim as f32,
                sigma: sigma,
            })
        } else {
            Err("Please check your dimension")
        }
    }
}

impl Kernel for LucyKernel {
    fn get_wij(&self, rij: f32, h: f32) -> f32 {
        // fraction of smoothing length
        let frac_1_h = 1. / h;
        let q = rij * frac_1_h;
        let tmp = if q > 1. {
            0.
        } else {
            (1. + 3. * q) * (1. - q).powf(3.)
        };
        let sigma_d = self.sigma * frac_1_h.powf(self.dim);

        sigma_d * tmp
    }

    fn get_dwij(&self, xij: &Vec<f32>, dwij: &mut Vec<f32>, rij: f32, h: f32) {
        if rij > 1e-12 {
            // compute the derivative of kernel w.r.t (x_i - x_j) or x_i
            let frac_1_h = 1. / h;
            let q = rij * frac_1_h;
            let tmp = if q > 2. {
                0.
            } else if q > 1. && q <= 2. {
                -0.75 * frac_1_h * (2. - q).powf(2.)
            } else {
                -0.75 * frac_1_h * (4. * q - 3. * q.powf(2.))
            };

            let dw_drij = tmp * self.sigma * frac_1_h.powf(self.dim);

            // xij is x_i - x_j
            dwij[0] = dw_drij * xij[0] * 1. / rij;
            dwij[1] = dw_drij * xij[1] * 1. / rij;
            dwij[2] = dw_drij * xij[2] * 1. / rij;
        } else {
            dwij[0] = 0.;
            dwij[1] = 0.;
            dwij[2] = 0.;
        }
    }
}

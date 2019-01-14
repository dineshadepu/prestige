use std::f32::consts::FRAC_1_PI;
use super::Kernel;

/// Look at Smoothed Particle Hydrodynamics A Study of the possibilities of SPH in hydraulic engineering
pub struct CubicKernel {
    pub dim: f32,
    pub sigma: f32,
}

impl CubicKernel {
    pub fn new(dim: usize) -> Result<CubicKernel, &'static str> {
        let tmp = match dim {
            1 => Some(2. / 3.),
            2 => Some(10. / 7. * FRAC_1_PI),
            3 => Some(FRAC_1_PI),
            _ => None,
        };

        if let Some(sigma) = tmp {
            Ok(CubicKernel {
                dim: dim as f32,
                sigma: sigma,
            })
        } else {
            Err("Prestige is only a 2d simulator, dimension should either be 1 or two")
        }
    }
}

impl Kernel for CubicKernel {
    fn get_wij(&self, rij: f32, h: f32) -> f32 {
        // fraction of smoothing length
        let frac_1_h = 1. / h;
        let q = rij * frac_1_h;
        let tmp = if q > 2. {
            0.
        } else if q > 1. && q <= 2. {
            0.25 * (2. - q).powf(3.)
        } else {
            1. - 1.5 * q.powf(2.) + 0.75 * q.powf(3.)
        };
        tmp * self.sigma * frac_1_h.powf(self.dim)
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
        }
        else{
            dwij[0] = 0.;
            dwij[1] = 0.;
            dwij[2] = 0.;
        }
    }
}

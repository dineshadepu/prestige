pub mod cubic_kernel;
pub mod lucy_kernel;
pub mod gaussian_kernel;

#[cfg(test)]
pub mod test_kernel;
#[cfg(test)]
pub mod test_cubic_kernel;
#[cfg(test)]
pub mod test_lucy_kernel;
#[cfg(test)]
pub mod test_gaussian_kernel;

pub trait Kernel {
    fn get_wij(&self, rij: f32, h: f32) -> f32;
    fn get_dwij(&self, xij: &Vec<f32>, dwij: &mut Vec<f32>, rij: f32, h: f32);
}

// reuse imports
pub use self::cubic_kernel::CubicKernel;

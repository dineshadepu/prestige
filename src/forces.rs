use crate::particles::Particles;
use cudarc::driver::{CudaFunction, CudaStream, DriverError, LaunchConfig, PushKernelArg};
use std::sync::Arc;

pub type GpuResult<T> = Result<T, DriverError>;

pub trait ForceModel {
    fn compute(&mut self, p: &mut Particles) -> GpuResult<()>;
}

pub struct GravityForce {
    g: f64,
    kernel: CudaFunction,
}

impl GravityForce {
    pub fn new(g: f64, kernel: CudaFunction) -> Self {
        Self { g, kernel }
    }
}

impl ForceModel for GravityForce {
    fn compute(&mut self, p: &mut Particles) -> GpuResult<()> {
        let cfg = LaunchConfig::for_num_elems(p.n as u32);
        let mut launch = p.stream().launch_builder(&self.kernel);
        launch.arg(&mut p.fy);
        launch.arg(&p.n);
        launch.arg(&self.g);
        unsafe { launch.launch(cfg) }?;
        Ok(())
    }
}

pub struct DEMParticleParticleForce {
    kn: f64,
    stream: Arc<CudaStream>,
    kernel: CudaFunction,
}

impl DEMParticleParticleForce {
    pub fn new(kn: f64, stream: Arc<CudaStream>, kernel: CudaFunction) -> Self {
        Self { kn, stream, kernel }
    }
}

impl ForceModel for DEMParticleParticleForce {
    fn compute(&mut self, p: &mut Particles) -> GpuResult<()> {
        let n = p.n;
        let fy = &mut p.fy;

        let cfg = LaunchConfig::for_num_elems(p.n as u32);
        let mut launch = p.stream().launch_builder(&self.kernel);
        launch.arg(fy);
        launch.arg(n);
        launch.arg(&self.kn); // FIXED (was g before)
        unsafe { launch.launch(cfg) }?;
        Ok(())
    }
}

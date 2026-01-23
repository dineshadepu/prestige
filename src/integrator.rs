use crate::particles::Particles;
use cudarc::driver::{CudaFunction, DriverError, LaunchConfig, PushKernelArg};

pub type GpuResult<T> = Result<T, DriverError>;

pub trait Integrator {
    fn stage1(&mut self, p: &mut Particles) -> GpuResult<()>;
    fn stage2(&mut self, p: &mut Particles) -> GpuResult<()>;
    fn stage3(&mut self, p: &mut Particles) -> GpuResult<()>;
}

pub struct LeapfrogIntegrator {
    dt: f64,
    kernel_stage1: CudaFunction,
    kernel_stage2: CudaFunction,
    kernel_stage3: CudaFunction,
}

impl LeapfrogIntegrator {
    pub fn new(
        dt: f64,
        kernel_stage1: CudaFunction,
        kernel_stage2: CudaFunction,
        kernel_stage3: CudaFunction,
    ) -> Self {
        Self {
            dt,
            kernel_stage1,
            kernel_stage2,
            kernel_stage3,
        }
    }
}

impl Integrator for LeapfrogIntegrator {
    fn stage1(&mut self, p: &mut Particles) -> GpuResult<()> {
        let cfg = LaunchConfig::for_num_elems(p.n as u32);
        let mut launch = p.stream().launch_builder(&self.kernel_stage1);
        launch.arg(&mut p.vy);
        launch.arg(&p.fy);
        launch.arg(&p.n);
        launch.arg(&self.dt);
        unsafe { launch.launch(cfg) }?;
        Ok(())
    }

    fn stage2(&mut self, p: &mut Particles) -> GpuResult<()> {
        let cfg = LaunchConfig::for_num_elems(p.n as u32);
        let mut launch = p.stream().launch_builder(&self.kernel_stage2);
        launch.arg(&mut p.vy);
        launch.arg(&p.fy);
        launch.arg(&p.n);
        launch.arg(&self.dt);
        unsafe { launch.launch(cfg) }?;
        Ok(())
    }

    fn stage3(&mut self, p: &mut Particles) -> GpuResult<()> {
        let cfg = LaunchConfig::for_num_elems(p.n as u32);
        let mut launch = p.stream().launch_builder(&self.kernel_stage3);
        launch.arg(&mut p.vy);
        launch.arg(&p.fy);
        launch.arg(&p.n);
        launch.arg(&self.dt);
        unsafe { launch.launch(cfg) }?;
        Ok(())
    }
}

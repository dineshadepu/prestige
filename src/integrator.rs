use crate::particles::Particles;
use cudarc::driver::CudaSlice;
use cudarc::driver::{CudaFunction, DriverError, LaunchConfig, PushKernelArg};

pub type GpuResult<T> = Result<T, DriverError>;

pub trait Integrator {
    fn stage1(&mut self, p: &mut Particles) -> GpuResult<()>;
    fn stage2(&mut self, p: &mut Particles) -> GpuResult<()>;
    fn stage3(&mut self, p: &mut Particles) -> GpuResult<()>;
}

pub struct LeapfrogIntegrator {
    dt: CudaSlice<f64>,      // device scalar
    half_dt: CudaSlice<f64>, // device scalar

    kernel_stage1: CudaFunction,
    kernel_stage2: CudaFunction,
    kernel_stage3: CudaFunction,
}

impl LeapfrogIntegrator {
    pub fn new(
        dt: f64,
        p: &Particles, // need stream
        kernel_stage1: CudaFunction,
        kernel_stage2: CudaFunction,
        kernel_stage3: CudaFunction,
    ) -> GpuResult<Self> {
        let half = 0.5 * dt;

        let dt_dev = p.stream.clone_htod(&[dt])?;
        let half_dt_dev = p.stream.clone_htod(&[half])?;

        Ok(Self {
            dt: dt_dev,
            half_dt: half_dt_dev,
            kernel_stage1,
            kernel_stage2,
            kernel_stage3,
        })
    }
}

impl Integrator for LeapfrogIntegrator {
    fn stage1(&mut self, p: &mut Particles) -> GpuResult<()> {
        let n = p.n_host[0] as u32;
        let cfg = LaunchConfig::for_num_elems(n);

        let Particles {
            u,
            force,
            m,
            n: n_dev,
            ..
        } = p;

        let stream = p.stream.clone();
        let mut launch = stream.launch_builder(&self.kernel_stage1);
        launch.arg(u);
        launch.arg(force);
        launch.arg(m);
        launch.arg(n_dev);
        launch.arg(&self.half_dt);
        unsafe { launch.launch(cfg) }?;
        Ok(())
    }

    fn stage2(&mut self, p: &mut Particles) -> GpuResult<()> {
        let n = p.n_host[0] as u32;
        let cfg = LaunchConfig::for_num_elems(n);

        let Particles { x, u, n: n_dev, .. } = p;

        let stream = p.stream.clone();
        let mut launch = stream.launch_builder(&self.kernel_stage2);
        launch.arg(x);
        launch.arg(u);
        launch.arg(n_dev);
        launch.arg(&self.dt);
        unsafe { launch.launch(cfg) }?;
        Ok(())
    }

    fn stage3(&mut self, p: &mut Particles) -> GpuResult<()> {
        let n = p.n_host[0] as u32;
        let cfg = LaunchConfig::for_num_elems(n);

        let Particles {
            u,
            force,
            m,
            n: n_dev,
            ..
        } = p;

        let stream = p.stream.clone();
        let mut launch = stream.launch_builder(&self.kernel_stage3);
        launch.arg(u);
        launch.arg(force);
        launch.arg(m);
        launch.arg(n_dev);
        launch.arg(&self.half_dt);
        unsafe { launch.launch(cfg) }?;
        Ok(())
    }
}

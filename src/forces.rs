// Copyright (c) 2026 Dinesh Adepu
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use crate::particles::Particles;
use cudarc::driver::{CudaFunction, CudaSlice, DriverError, LaunchConfig, PushKernelArg};

pub type GpuResult<T> = Result<T, DriverError>;

pub trait ForceModel {
    fn compute(&mut self, p: &mut Particles) -> GpuResult<()>;
}

pub struct ResetForce {
    kernel: CudaFunction,
}

impl ResetForce {
    pub fn new(p: &Particles, kernel: CudaFunction) -> GpuResult<Self> {
        Ok(Self { kernel })
    }
}

impl ForceModel for ResetForce {
    fn compute(&mut self, p: &mut Particles) -> GpuResult<()> {
        let n_host = p.n_host[0];

        let Particles { force, n, .. } = p;

        let cfg = LaunchConfig::for_num_elems(n_host as u32);
        let stream = p.stream.clone();
        let mut launch = stream.launch_builder(&self.kernel);
        launch.arg(force);
        launch.arg(n);
        unsafe { launch.launch(cfg) }?;
        Ok(())
    }
}

pub struct GravityForce {
    g: CudaSlice<f64>, // length = 3
    kernel: CudaFunction,
}

impl GravityForce {
    pub fn new(g: [f64; 3], p: &Particles, kernel: CudaFunction) -> GpuResult<Self> {
        let g_dev = p.stream.clone_htod(&g)?;
        Ok(Self { g: g_dev, kernel })
    }
}

impl ForceModel for GravityForce {
    fn compute(&mut self, p: &mut Particles) -> GpuResult<()> {
        let n_host = p.n_host[0];

        let Particles { force, m, n, .. } = p;

        let cfg = LaunchConfig::for_num_elems(n_host as u32);
        let stream = p.stream.clone();
        let mut launch = stream.launch_builder(&self.kernel);
        launch.arg(force);
        launch.arg(m);
        launch.arg(&self.g);
        launch.arg(n);
        unsafe { launch.launch(cfg) }?;
        Ok(())
    }
}

pub struct DEMParticleParticleForce {
    kn: CudaSlice<f64>,          // device scalar
    cor_pp: CudaSlice<f64>,      // device scalar
    friction_pp: CudaSlice<f64>, // device scalar
    kernel: CudaFunction,
}

impl DEMParticleParticleForce {
    pub fn new(
        kn: f64,
        cor_pp: f64,
        friction_pp: f64,
        p: &Particles,
        kernel: CudaFunction,
    ) -> GpuResult<Self> {
        let kn_dev = p.stream.clone_htod(&[kn])?;
        let cor_dev = p.stream.clone_htod(&[cor_pp])?;
        let fric_dev = p.stream.clone_htod(&[friction_pp])?;

        Ok(Self {
            kn: kn_dev,
            cor_pp: cor_dev,
            friction_pp: fric_dev,
            kernel,
        })
    }
}

impl ForceModel for DEMParticleParticleForce {
    fn compute(&mut self, p: &mut Particles) -> GpuResult<()> {
        let n_host = p.n_host[0];

        let Particles {
            x,
            u,
            force,
            m,
            rad,
            n,
            ..
        } = p;

        let cfg = LaunchConfig::for_num_elems(n_host as u32);
        let stream = p.stream.clone();
        let mut launch = stream.launch_builder(&self.kernel);

        launch.arg(x);
        launch.arg(u);
        launch.arg(force);
        launch.arg(m);
        launch.arg(rad);
        launch.arg(n);
        launch.arg(&self.kn);
        launch.arg(&self.cor_pp);
        launch.arg(&self.friction_pp);

        unsafe { launch.launch(cfg) }?;
        Ok(())
    }
}

pub struct FreezeBdryForce {
    kernel: CudaFunction,
}

impl FreezeBdryForce {
    pub fn new(p: &Particles, kernel: CudaFunction) -> GpuResult<Self> {
        Ok(Self { kernel })
    }
}

impl ForceModel for FreezeBdryForce {
    fn compute(&mut self, p: &mut Particles) -> GpuResult<()> {
        let n_host = p.n_host[0];

        let Particles {
            force,
            body_type,
            n,
            ..
        } = p;

        let cfg = LaunchConfig::for_num_elems(n_host as u32);
        let stream = p.stream.clone();
        let mut launch = stream.launch_builder(&self.kernel);
        launch.arg(force);
        launch.arg(body_type);
        launch.arg(n);
        unsafe { launch.launch(cfg) }?;
        Ok(())
    }
}

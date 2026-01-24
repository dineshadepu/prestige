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

use crate::DIM;
use cudarc::driver::{CudaSlice, CudaStream, DriverError};
use std::fs::File;
use std::io::Write;
use std::sync::Arc;

pub type GpuResult<T> = Result<T, DriverError>;

pub struct Particles {
    pub n_host: Vec<u32>,  // host scalar
    pub n: CudaSlice<u32>, // device scalar

    // vec_double_type -> flattened [N * DIM]
    pub x: CudaSlice<f64>,
    pub u: CudaSlice<f64>,
    pub force: CudaSlice<f64>,

    // double_type
    pub m: CudaSlice<f64>,
    pub rad: CudaSlice<f64>,

    // Body type (to differentiate if it is a boundary)
    pub body_type: CudaSlice<f64>,

    pub stream: Arc<CudaStream>,
}

impl Particles {
    pub fn new(n: u32, stream: Arc<CudaStream>) -> GpuResult<Self> {
        let n_host = vec![n];
        let n_dev = stream.clone_htod(&[n])?;

        let n_usize = n as usize;

        Ok(Self {
            n_host,
            n: n_dev,

            x: stream.alloc_zeros::<f64>(n_usize * DIM)?,
            u: stream.alloc_zeros::<f64>(n_usize * DIM)?,
            force: stream.alloc_zeros::<f64>(n_usize * DIM)?,

            m: stream.alloc_zeros::<f64>(n_usize)?,
            rad: stream.alloc_zeros::<f64>(n_usize)?,

            body_type: stream.alloc_zeros::<f64>(n_usize)?,

            stream,
        })
    }
    pub fn write_vtk(&self, step: usize) -> std::io::Result<()> {
        let x_host = self.stream.clone_dtoh(&self.x).unwrap();
        let n = x_host.len() / 3;

        let fname = format!("out_{:06}.vtk", step);
        let mut f = File::create(fname)?;

        writeln!(f, "# vtk DataFile Version 3.0")?;
        writeln!(f, "DEM particles")?;
        writeln!(f, "ASCII")?;
        writeln!(f, "DATASET POLYDATA")?;
        writeln!(f, "POINTS {} double", n)?;

        for i in 0..n {
            let k = 3 * i;
            writeln!(f, "{} {} {}", x_host[k], x_host[k + 1], x_host[k + 2])?;
        }

        Ok(())
    }
}

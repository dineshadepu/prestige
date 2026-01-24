use crate::DIM;
use cudarc::driver::{CudaSlice, CudaStream, DriverError};
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
}

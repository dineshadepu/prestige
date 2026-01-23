use cudarc::driver::{CudaSlice, CudaStream, DriverError};
use std::sync::Arc;

pub type GpuResult<T> = Result<T, DriverError>;

pub struct Particles {
    pub n: CudaSlice<i64>,

    pub x: CudaSlice<f64>,
    pub y: CudaSlice<f64>,
    pub vx: CudaSlice<f64>,
    pub vy: CudaSlice<f64>,
    pub fy: CudaSlice<f64>,

    stream: Arc<CudaStream>,
}

impl Particles {
    pub fn new(n: i32, stream: Arc<CudaStream>) -> GpuResult<Self> {
        Ok(Self {
            n: stream.alloc_zeros::<f64>(n as usize)?,
            x: stream.alloc_zeros::<f64>(n as usize)?,
            y: stream.alloc_zeros::<f64>(n as usize)?,
            vx: stream.alloc_zeros::<f64>(n as usize)?,
            vy: stream.alloc_zeros::<f64>(n as usize)?,
            fy: stream.alloc_zeros::<f64>(n as usize)?,
            stream,
        })
    }

    pub fn stream(&self) -> &CudaStream {
        &self.stream
    }
}

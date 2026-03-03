use crate::particles::Particles;
use crate::cuda::GpuResult;
use cudarc::driver::DevicePtr;
// use cudarc::driver::{CudaFunction, LaunchConfig, PushKernelArg};
use cudarc::driver::{CudaSlice, CudaStream, DriverError};
use std::ffi::c_void;
use std::sync::Arc;


#[repr(C)]
#[derive(Clone, Copy)]
pub struct Double3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

unsafe impl cudarc::driver::DeviceRepr for Double3 {}

pub type GridHandle = *mut c_void;

unsafe extern "C" {
    pub fn grid_create() -> GridHandle;
    pub fn grid_destroy(grid: GridHandle);

    pub fn grid_config(
        grid: GridHandle,
        cell_x: i32,
        cell_y: i32,
        cell_z: i32,
        inv_h: f64,
        domain_min: Double3,
        domain_max: Double3,
    );
    pub fn grid_build(
        grid: GridHandle,
        d_pos: *const f64,
        np: i32,
        part_idx: *mut i32,
        cell_beg: *mut i32,
        cell_end: *mut i32,
    );

    pub fn grid_cell_begin(grid: GridHandle) -> *const i32;
    pub fn grid_cell_end(grid: GridHandle) -> *const i32;
    pub fn grid_particle_perm(grid: GridHandle) -> *const i32;
}

pub struct Grid {
    pub handle: GridHandle,

    pub part_idx: CudaSlice<i32>,
    pub cell_beg: CudaSlice<i32>,
    pub cell_end: CudaSlice<i32>,

    pub cell_num_x: i32,
    pub cell_num_y: i32,
    pub cell_num_z: i32,
    pub inv_h: f64,
    pub domain_min: Double3,
    pub domain_max: Double3,
}

impl Grid {
    pub fn new(
        stream: Arc<CudaStream>,
        np: usize,
        ncell: usize,
        cell_num_x: i32,
        cell_num_y: i32,
        cell_num_z: i32,
        inv_h: f64,
        domain_min: Double3,
        domain_max: Double3,
    ) -> GpuResult<Self> {
        let part_idx = stream.alloc_zeros::<i32>(np)?;
        let cell_beg = stream.alloc_zeros::<i32>(ncell)?;
        let cell_end = stream.alloc_zeros::<i32>(ncell)?;

        let handle = unsafe { grid_create() };

        unsafe {
            grid_config(
                handle, cell_num_x, cell_num_y, cell_num_z, inv_h, domain_min, domain_max,
            );
        }

        Ok(Self {
            handle,
            part_idx,
            cell_beg,
            cell_end,
            cell_num_x,
            cell_num_y,
            cell_num_z,
            inv_h,
            domain_min,
            domain_max,
        })
    }
    pub fn build(&self, particles: &Particles) -> GpuResult<()> {
        unsafe {
            let stream = &particles.stream;

            let (x_ptr, _sync) = particles.x.device_ptr(stream);
            let (idx_ptr, _) = self.part_idx.device_ptr(stream);
            let (beg_ptr, _) = self.cell_beg.device_ptr(stream);
            let (end_ptr, _) = self.cell_end.device_ptr(stream);

            grid_build(
                self.handle,
                x_ptr as *const f64,
                particles.n_host[0] as i32,
                idx_ptr as *mut i32,
                beg_ptr as *mut i32,
                end_ptr as *mut i32,
            );
        }
        Ok(())
    }
}

impl Drop for Grid {
    fn drop(&mut self) {
        unsafe { grid_destroy(self.handle) }
    }
}

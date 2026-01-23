use cudarc::driver::{CudaContext, CudaFunction, CudaStream, DriverError};
use cudarc::nvrtc::Ptx;
use std::sync::Arc;

pub type GpuResult<T> = Result<T, DriverError>;

pub struct CudaManager {
    ctx: Arc<CudaContext>,
}

impl CudaManager {
    pub fn new(device: usize) -> GpuResult<Self> {
        let ctx = CudaContext::new(device)?;
        Ok(Self { ctx: Arc::new(ctx) })
    }

    pub fn new_stream(&self) -> GpuResult<Arc<CudaStream>> {
        self.ctx.new_stream()
    }

    pub fn load_module(&self, ptx_path: &str) -> GpuResult<ModuleHandle> {
        let ptx = Ptx::from_file(ptx_path);
        let module = self.ctx.load_module(ptx)?;
        Ok(ModuleHandle { module })
    }
}

pub struct ModuleHandle {
    module: cudarc::driver::CudaModule,
}

impl ModuleHandle {
    pub fn get(&self, name: &str) -> GpuResult<CudaFunction> {
        Ok(self.module.get_func(name)?)
    }
}

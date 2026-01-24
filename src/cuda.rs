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

use cudarc::driver::*;
use std::sync::Arc;

pub type GpuResult<T> = Result<T, DriverError>;

pub struct CudaManager {
    pub ctx: Arc<CudaContext>,
}

impl CudaManager {
    pub fn new(device: usize) -> GpuResult<Self> {
        let ctx = CudaContext::new(device)?; // already Arc
        Ok(Self { ctx })
    }

    pub fn new_stream(&self) -> GpuResult<Arc<CudaStream>> {
        Ok(self.ctx.new_stream()?)
    }

    pub fn load_module(&self, path: &str) -> GpuResult<ModuleHandle> {
        let ptx = std::fs::read_to_string(path).expect("Failed to read PTX file");
        let module = self.ctx.load_module(ptx.into())?;
        Ok(ModuleHandle { module })
    }
}

pub struct ModuleHandle {
    pub module: Arc<CudaModule>,
}

impl ModuleHandle {
    pub fn get(&self, name: &str) -> GpuResult<CudaFunction> {
        Ok(self.module.load_function(name)?)
    }
}

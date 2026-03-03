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

pub mod cuda;
pub mod forces;
pub mod integrator;
pub mod particles;
pub mod grid;

pub const DIM: usize = 3;
pub const MAX_NO_WALLS: usize = 8;
pub const MAX_CNTS: usize = 32;

pub use cuda::*;
pub use forces::*;
pub use integrator::*;
pub use particles::*;
pub use grid::*;

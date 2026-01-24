pub mod cuda;
pub mod forces;
pub mod integrator;
pub mod particles;

pub const DIM: usize = 3;
pub const MAX_NO_WALLS: usize = 8;
pub const MAX_CNTS: usize = 32;

pub use cuda::*;
pub use forces::*;
pub use integrator::*;
pub use particles::*;

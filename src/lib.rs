// see this for better understanding
// https://doc.rust-lang.org/book/first-edition/macros.html
#[macro_use]
pub mod contact_search;

pub mod physics;

// basic utilities

// Output trait for every particle array
pub trait WriteOutput {
    fn write_vtk(&self, output: String);
}

// Integrator trait
pub trait RK2Integrator {
    fn initialize(&mut self);
    fn stage1(&mut self, dt: f32);
    fn stage2(&mut self, dt: f32);
}

trait EulerIntegrator {
    fn stage1(&mut self);
}

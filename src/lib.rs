#[cfg(test)]
extern crate fluid;
extern crate rayon; // 1.0.3
extern crate simple_shapes;

pub mod physics;
pub mod contact_search;

// basic utilities

// Output trait for every particle array
pub trait WriteOutput{
    fn write_vtk(&self, output: String);
}

// Integrator trait
pub trait RK2Integrator{
    fn initialize(&mut self);
    fn stage1(&mut self, dt: f32);
    fn stage2(&mut self, dt: f32);
}

trait EulerIntegrator{
    fn stage1(&mut self);
}

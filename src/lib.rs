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

pub trait EulerIntegrator {
    fn stage1(&mut self, dt: f32);
}

pub fn rk2_initialize<T: RK2Integrator>(entites: &mut Vec<&mut T>){
    for entity in entites{
        entity.initialize();
    }
}

pub fn rk2_stage_1<T: RK2Integrator>(entites: &mut Vec<&mut T>, dt: f32){
    for entity in entites{
        entity.stage1(dt);
    }
}

pub fn rk2_stage_2<T: RK2Integrator>(entites: &mut Vec<&mut T>, dt: f32){
    for entity in entites{
        entity.stage2(dt);
    }
}

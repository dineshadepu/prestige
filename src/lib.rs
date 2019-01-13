// see this for better understanding
// https://doc.rust-lang.org/book/first-edition/macros.html
#[macro_use]
pub mod contact_search;

pub mod geometry;
pub mod physics;

// external imports
use indicatif::{ProgressBar, ProgressStyle};
// basic utilities

// Output trait for every particle array
pub trait WriteOutput {
    fn write_vtk(&self, output: String);
}

// Integrator trait
pub trait RK2Integrator {
    fn rk2_initialize(&mut self);
    fn rk2_stage_1(&mut self, dt: f32);
    fn rk2_stage_2(&mut self, dt: f32);
}

pub trait EulerIntegrator {
    fn euler_stage_1(&mut self, dt: f32);
}

pub fn rk2_initialize<T: RK2Integrator>(entites: &mut Vec<&mut T>) {
    for entity in entites {
        entity.rk2_initialize();
    }
}

pub fn rk2_stage_1<T: RK2Integrator>(entites: &mut Vec<&mut T>, dt: f32) {
    for entity in entites {
        entity.rk2_stage_1(dt);
    }
}

pub fn rk2_stage_2<T: RK2Integrator>(entites: &mut Vec<&mut T>, dt: f32) {
    for entity in entites {
        entity.rk2_stage_2(dt);
    }
}

pub fn setup_progress_bar(total_steps: u64) -> ProgressBar {
    let pb = ProgressBar::new(total_steps);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] ({eta})")
            .progress_chars("#>-"),
    );
    pb
}

pub fn print_no_part(pars: Vec<&Vec<f32>>) {
    let mut total_pars = 0;
    for x in pars {
        total_pars += x.len();
    }
    println!("Total particles {}", total_pars);
}

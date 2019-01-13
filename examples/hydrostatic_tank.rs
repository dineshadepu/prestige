extern crate prestige;
extern crate indicatif;

// crates imports
use prestige::{
    contact_search::linked_nnps::{stash_2d, WorldBounds, LinkedNNPS},
    EulerIntegrator, WriteOutput,
    physics::rigid_body::{RB3d, equations::{linear_interparticle_force, apply_gravity}}
    // physics::sph::wcsph,
};

// external crate imports
use indicatif::{ProgressBar, ProgressStyle};
use prestige::geometry::{grid_arange_3d, tank_3d};

// std imports
use std::fs;


fn create_entites(spacing : f32) -> (RB3d, RB3d){
    // Create the cube and the tank
    let (xc, yc, zc) = grid_arange_3d(1., 2., spacing, 0.5, 1.5, spacing, 1., 2., spacing);
    let rc = vec![spacing / 2.; xc.len()];
    let mut cube = RB3d::from_xyzr(xc.clone(), yc, zc, rc);
    let rho_c = 2000.;
    let mc = rho_c * spacing.powf(3.);
    cube.m = vec![mc; xc.len()];
    cube.nnps_idx = 0;
    let (xt, yt, zt) = tank_3d(-1.1, 3., spacing, -1.1, 3., spacing, -1., 3., spacing, 2);
    let rt = vec![spacing / 2.; xt.len()];
    let mut tank = RB3d::from_xyzr(xt.clone(), yt, zt, rt);
    let rho_t = 2000.;
    let mt = rho_t * spacing.powf(3.);
    tank.m = vec![mt; xt.len()];
    tank.nnps_idx = 1;

    // Compute the predefined quantities
    cube.initialize();
    tank.initialize();

    (cube, tank)
}

fn print_no_part(pars: Vec<&Vec<f32>>){
    let mut total_pars = 0;
    for x in pars{
        total_pars += x.len();
    }
    println!("Total particles {}", total_pars);
}


fn main() {
    let spacing = 0.1;
    // dimension
    let dim = 3;

    // particles
    let (mut body, tank) = create_entites(spacing);

    let kn = 1e5;

    print_no_part(vec![&body.x, &tank.x]);

    // setup nnps
    let world_bounds = WorldBounds::new(-1.1, 3.1, -1.1, 4.1, -1.0, 4.0, spacing);
    let mut nnps = LinkedNNPS::new(2, &world_bounds, dim);

    // solver data
    let dt = 1e-4;
    let mut t = 0.;
    let tf = 1.;
    let mut step_no = 0;
    let pfreq = 10;

    let project_root = env!("CARGO_MANIFEST_DIR");
    let dir_name = project_root.to_owned() + "/rb_1_output";
    let _p = fs::create_dir(&dir_name);

    // create a progress bar
    let total_steps = (tf / dt) as u64;
    let pb = ProgressBar::new(total_steps);
    pb.set_style(ProgressStyle::default_bar()
                 .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] ({eta})")
                 .progress_chars("#>-"));
    while t < tf {
        // stash the particles into the world's cells
        stash_2d(vec![&body, &tank], &mut nnps);

        apply_gravity(&body.m, &mut body.fx, &mut body.fy, &mut body.fz, 0.0, -9.81, 0.0);
        linear_interparticle_force(
            &body.x, &body.y, &body.z,
            &body.u, &body.v, &body.w, &body.rad,
            &mut body.fx, &mut body.fy, &mut body.fz,

            &tank.x, &tank.y, &tank.z,
            &tank.u, &tank.v, &tank.w, &tank.rad, tank.nnps_idx,

            &nnps, kn, 5.
        );

        body.euler_stage_1(dt);

        if step_no % pfreq == 0 {
            tank.write_vtk(format!("{}/tank_{}.vtk", &dir_name, step_no));
            body.write_vtk(format!("{}/body_{}.vtk", &dir_name, step_no));
        }
        step_no += 1;
        t += dt;

        // progress bar increment
        pb.inc(1);
    }
    pb.finish_with_message("Simulation succesfully completed");
}

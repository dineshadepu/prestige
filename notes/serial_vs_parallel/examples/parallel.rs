extern crate indicatif;
extern crate prestige;
extern crate simple_shapes;

// external crate imports
use indicatif::{ProgressBar, ProgressStyle};

// crates imports
use prestige::{
    contact_search::{NNPS, WorldBounds, stash_2d},
    physics::dem::{
        equations::{body_force, contact_force_par},
        DEM,
    },
    RK2Integrator, WriteOutput,
};
use simple_shapes::{grid_arange, tank};

// std imports
use std::{fs, env};

fn main() {
    let args: Vec<_> = env::args().collect();
    if args.len() < 2 {
        println!("Please give the spacing between the particles");
    }

    let spacing = args[1].parse::<f32>().unwrap();
    let (xt, yt) = tank(-1., 3., spacing, -1., 4., spacing, 1);
    let (xb, yb) = grid_arange(
        -0.9,
        0.1 + spacing / 2.,
        spacing,
        0.0,
        1.0 + spacing / 2.,
        spacing,
    );
    let yb = yb.iter().map(|i| i - 0.95).collect::<Vec<f32>>();
    let body_particle_no = xb.len();
    let tank_particle_no = xt.len();

    // define density of the particle
    let rho_b = 2000.;
    let m_single_par = rho_b * spacing.powf(2.);
    let mut body = DEM::new_from_xyzh(
        xb.clone(),
        yb.clone(),
        vec![1.2 * spacing; body_particle_no],
        body_particle_no,
    );
    body.m = vec![m_single_par; body_particle_no];
    body.r = vec![spacing / 4.; body_particle_no];
    body.nnps_idx = 0;

    let mut tank = DEM::new_from_xyzh(
        xt.clone(),
        yt.clone(),
        vec![1.2 * spacing; tank_particle_no],
        tank_particle_no,
    );
    tank.m = vec![m_single_par; tank_particle_no];
    tank.r = vec![spacing / 2.; tank_particle_no];
    tank.nnps_idx = 1;
    println!("Radius of body is {}", body.r[0]);
    println!("Radius of tank is {}", tank.r[0]);

    let kn = 1e5;
    let dim = 2;

    println!(
        "Body particles: {}, tank particles: {}, Total particles: {}",
        body.x.len(),
        tank.x.len(),
        body.x.len() + tank.x.len()
    );

    // setup nnps
    let world_bounds = WorldBounds::new(-1.1, 3.1, -1.1, 4.1, 0.0, 0.0, spacing);
    let mut nnps = NNPS::new(2, &world_bounds, dim);

    // solver data
    let dt = 1e-4;
    let mut t = 0.;
    let tf = 10. * dt;
    let mut step_no = 0;
    let pfreq = 100;

    let version = env!("CARGO_MANIFEST_DIR");
    let dir_name = version.to_owned() + "/parallel_dam_break_output";

    match fs::create_dir(&dir_name){
        Ok(()) => (),
        Err(_msg) => (),
    }

    // create a progressbar
    let total_steps = (tf / dt) as u64;
    println!("\n \n");
    let pb = ProgressBar::new(total_steps);
    pb.set_style(ProgressStyle::default_bar()
                 .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({eta})")
                 .progress_chars("#>-"));

    while t < tf {
        // stash the particles into the world's cells
        stash_2d(vec![&body, &tank], &mut nnps);

        body.initialize();

        // stage 1
        body_force(&mut body.fx, &mut body.fy, &body.m, 0.0, -9.81);
        contact_force_par(
            &body.x,
            &body.y,
            &body.z,
            &body.r,
            &mut body.fx,
            &mut body.fy,
            &mut body.fz,
            &tank.x,
            &tank.y,
            &tank.z,
            &tank.r,
            tank.nnps_idx,
            &nnps,
            kn,
            dim,
        );
        contact_force_par(
            &body.x,
            &body.y,
            &body.z,
            &body.r,
            &mut body.fx,
            &mut body.fy,
            &mut body.fz,
            &body.x,
            &body.y,
            &body.z,
            &body.r,
            body.nnps_idx,
            &nnps,
            kn,
            dim,
        );

        body.stage1(dt);

        // stage 2
        body_force(&mut body.fx, &mut body.fy, &body.m, 0.0, -9.81);
        contact_force_par(
            &body.x,
            &body.y,
            &body.z,
            &body.r,
            &mut body.fx,
            &mut body.fy,
            &mut body.fz,
            &tank.x,
            &tank.y,
            &tank.z,
            &tank.r,
            tank.nnps_idx,
            &nnps,
            kn,
            dim,
        );
        contact_force_par(
            &body.x,
            &body.y,
            &body.z,
            &body.r,
            &mut body.fx,
            &mut body.fy,
            &mut body.fz,
            &body.x,
            &body.y,
            &body.z,
            &body.r,
            body.nnps_idx,
            &nnps,
            kn,
            dim,
        );


        body.stage2(dt);

        if step_no % pfreq == 0 {
            // println!("{}", step_no);
            let filename = format!("{}/tank_{}.vtk", &dir_name, step_no);
            tank.write_vtk(filename);

            let filename = format!("{}/body_{}.vtk", &dir_name, step_no);
            body.write_vtk(filename);
            // println!("{} ", step_no);
        }
        step_no += 1;
        t += dt;
        // progressbar increment
        pb.inc(1);
    }
    pb.finish_with_message("Simulation succesfully completed");
}

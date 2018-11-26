extern crate indicatif;
extern crate prestige;
extern crate simple_shapes;

// external crate imports
use indicatif::{ProgressBar, ProgressStyle};

// crates imports
use prestige::{
    contact_search::stash,
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
    if args.len() < 1 {
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
    let body_particle_no = xb.len();
    let tank_particle_no = xt.len();

    // define density of the particle
    let rho_b = 2000.;
    let m_single_par = rho_b * spacing.powf(2.);
    let mut body = DEM {
        x: xb.clone(),
        y: yb.clone(),
        x0: xb.clone(),
        y0: yb.clone(),
        u: vec![0.; xb.len()],
        v: vec![-3.; xb.len()],
        u0: vec![0.; xb.len()],
        v0: vec![-3.; xb.len()],
        r: vec![spacing / 2.; body_particle_no],
        fx: vec![0.; body_particle_no],
        fy: vec![0.; body_particle_no],
        h: vec![spacing / 2.; body_particle_no],
        nnps_idx: 0,
        no_par: body_particle_no,
        m: vec![m_single_par; body_particle_no],
    };

    let tank = DEM {
        x: xt.clone(),
        y: yt.clone(),
        x0: xt.clone(),
        y0: yt.clone(),
        u: vec![0.; xt.len()],
        v: vec![0.; xt.len()],
        u0: vec![0.; xt.len()],
        v0: vec![0.; xt.len()],
        r: vec![spacing / 2.; tank_particle_no],
        fx: vec![0.; tank_particle_no],
        fy: vec![0.; tank_particle_no],
        h: vec![spacing / 2.; tank_particle_no],
        nnps_idx: 1,
        no_par: xt.len(),
        // Use mass same as body
        m: vec![m_single_par; tank_particle_no],
    };
    let kn = 1e7;

    println!(
        "Body particles: {}, tank particles: {}, Total particles: {}",
        body.x.len(),
        tank.x.len(),
        body.x.len() + tank.x.len()
    );

    // solver data
    let dt = 1e-4;
    let mut t = 0.;
    let tf = 10. * dt;
    let mut step_no = 0;
    let pfreq = 100;

    let version = env!("CARGO_MANIFEST_DIR");
    let dir_name = version.to_owned() + "/parallel_dam_break_output";
    let _tmp = fs::create_dir(&dir_name);

    // create a progressbar
    let total_steps = (tf / dt) as u64;
    println!("\n \n");
    let pb = ProgressBar::new(total_steps);
    pb.set_style(ProgressStyle::default_bar()
                 .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({eta})")
                 .progress_chars("#>-"));

    while t < tf {
        // stash the particles into the world's cells
        let nnps = stash(vec![&body, &tank]);

        body.initialize();

        // stage 1
        body_force(&mut body.fx, &mut body.fy, &body.m, 0.0, -9.81);
        contact_force_par(
            &body.x,
            &body.y,
            &body.r,
            &mut body.fx,
            &mut body.fy,
            &tank.x,
            &tank.y,
            &tank.r,
            tank.nnps_idx,
            &nnps,
            kn,
        );
        contact_force_par(
            &body.x,
            &body.y,
            &body.r,
            &mut body.fx,
            &mut body.fy,
            &body.x,
            &body.y,
            &body.r,
            body.nnps_idx,
            &nnps,
            kn,
        );

        body.stage1(dt);

        // stage 2
        body_force(&mut body.fx, &mut body.fy, &body.m, 0.0, -9.81);
        contact_force_par(
            &body.x,
            &body.y,
            &body.r,
            &mut body.fx,
            &mut body.fy,
            &tank.x,
            &tank.y,
            &tank.r,
            tank.nnps_idx,
            &nnps,
            kn,
        );
        contact_force_par(
            &body.x,
            &body.y,
            &body.r,
            &mut body.fx,
            &mut body.fy,
            &body.x,
            &body.y,
            &body.r,
            body.nnps_idx,
            &nnps,
            kn,
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

extern crate prestige;

// crates imports
use prestige::{
    contact_search::{stash_2d, WorldBounds, NNPS},
    physics::rigid_body::{
        equations::{apply_gravity, linear_interparticle_force},
        RB3d,
    },
    setup_progress_bar, EulerIntegrator, WriteOutput,
};

// external crate imports
use prestige::geometry::benchmarks::create_zhang_geometry;

// std imports
use std::fs;


fn create_entites(spacing: f32) -> (RB3d, RB3d){
    let (xc, yc, body_id, xt, yt) = create_zhang_geometry(spacing);

    // create and setup cylinders
    let mut cylinders = RB3d::from_xyr_b_id(xc.clone(), yc, vec![spacing/2.; xc.len()], body_id);
    let cylinder_rho = 2700.;
    let cylinder_m = cylinder_rho * spacing.powf(2.);
    // set the mass
    cylinders.m = vec![cylinder_m; cylinders.x.len()];
    cylinders.nnps_idx = 0;
    cylinders.initialize();

    // create and setup tank
    let mut tank = RB3d::from_xyr(xt.clone(), yt, vec![spacing / 2.; xt.len()]);
    let tank_rho = 1051.;
    let tank_m = tank_rho * spacing.powf(2.);

    // set the mass
    tank.m = vec![tank_m; tank.x.len()];
    tank.nnps_idx = 1;
    tank.initialize();

    (cylinders, tank)
}

fn print_no_part(pars: Vec<&Vec<f32>>) {
    let mut total_pars = 0;
    for x in pars {
        total_pars += x.len();
    }
    println!("Total particles {}", total_pars);
}

fn main() {
    // The diameter of the cylinder is 1 cm, which is 0.01 m. Let's the spacing be
    // 0.05 cm that would be 5 * 1e-5 m.
    let spacing = 1e-3;
    // dimension
    let dim = 2;

    // particles
    let (mut cylinders, tank) = create_entites(spacing);

    let kn = 1e5;

    print_no_part(vec![&cylinders.x, &tank.x]);

    // setup nnps
    let world_bounds = WorldBounds::new(-0.01, 0.3, -0.01, 0.3, 0.0, 0.0, 2. * spacing);
    let mut nnps = NNPS::new(2, &world_bounds, dim);

    // solver data
    let dt = 1e-4;
    let mut t = 0.;
    let tf = 1.0;
    let mut step_no = 0;
    let pfreq = 100;

    let project_root = env!("CARGO_MANIFEST_DIR");
    let dir_name = project_root.to_owned() + "/rb_multi_euler_output";
    let _p = fs::create_dir(&dir_name);

    // create a progress bar
    let total_steps = (tf / dt) as u64;
    let pb = setup_progress_bar(total_steps);
    while t < tf {
        // stash the particles into the world's cells
        stash_2d(vec![&cylinders, &tank], &mut nnps);

        apply_gravity(
            &cylinders.m, &mut cylinders.fx, &mut cylinders.fy, &mut cylinders.fz,
            0.0, -9.81, 0.0,
        );

        linear_interparticle_force(
            &cylinders.x, &cylinders.y, &cylinders.z,
            &cylinders.u, &cylinders.v, &cylinders.w, &cylinders.rad,
            &mut cylinders.fx, &mut cylinders.fy, &mut cylinders.fz,

            &cylinders.x, &cylinders.y, &cylinders.z,
            &cylinders.u, &cylinders.v, &cylinders.w, &cylinders.rad,
            cylinders.nnps_idx,

            &nnps, kn, 5.
        );

        linear_interparticle_force(
            &cylinders.x, &cylinders.y, &cylinders.z,
            &cylinders.u, &cylinders.v, &cylinders.w, &cylinders.rad,
            &mut cylinders.fx, &mut cylinders.fy, &mut cylinders.fz,

            &tank.x, &tank.y, &tank.z,
            &tank.u, &tank.v, &tank.w, &tank.rad, tank.nnps_idx,

            &nnps, kn, 5.
        );
        cylinders.euler_stage_1(dt);

        if step_no % pfreq == 0 {
            tank.write_vtk(format!("{}/tank_{}.vtk", &dir_name, step_no));
            cylinders.write_vtk(format!("{}/cylinders_{}.vtk", &dir_name, step_no));
        }
        step_no += 1;
        t += dt;

        // progress bar increment
        pb.inc(1);
    }
    pb.finish_with_message("Simulation succesfully completed");
}

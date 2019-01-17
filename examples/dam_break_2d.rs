extern crate prestige;

// crates imports
use prestige::{
    continuity_and_momentum_eq_macro,
    continuity_eq_macro,
    momentum_eq_macro,
    contact_search::linked_nnps::{stash_2d, LinkedNNPS, WorldBounds},
    physics::sph::{
        wcsph,
        kernel::CubicKernel,
        wcsph::equations::{
            continuity_equation,
            momentum_equation,
            continuity_and_momentum_equation, tait_equation,
            reset_wcsph_entity,
        },
        wcsph::WCSPH,
    },
    print_no_part, setup_progress_bar, RK2Integrator, WriteOutput,
};


use prestige::geometry::benchmarks::create_2d_breaking_dam_geometry;

// std imports
use std::fs;

fn create_entites(spacing: f32) -> (WCSPH, WCSPH) {
    // Get the dam break geometry
    let (xc, yc, xt, yt) = create_2d_breaking_dam_geometry(Some(spacing));
    let mut fluid = WCSPH::new_with_xy(xc.clone(), yc, 0);
    let mc = 1000. * spacing.powf(2.);
    fluid.rho = vec![1000.; xc.len()];
    fluid.m = vec![mc; xc.len()];
    fluid.h = vec![1.2 * spacing; xc.len()];
    let mut tank = WCSPH::new_with_xy(xt.clone(), yt, 1);
    let rho_t = 1000.;
    let mt = rho_t * spacing.powf(3.);
    tank.m = vec![mt; xt.len()];
    tank.h = vec![1.2 * spacing; xt.len()];
    tank.rho = vec![1000.; xt.len()];

    (fluid, tank)
}

fn main() {
    // simulation variables
    let c0 = (2. as f32 * 9.81 * 0.05).sqrt();
    let spacing = 0.003;
    let dim = 2;

    let (mut fluid, mut tank) = create_entites(spacing);

    print_no_part(vec![&fluid.x, &tank.x]);

    // setup nnps
    let world_bounds = WorldBounds::new(
        -0.1,
        0.21,
        -0.05,
        0.21,
        0.,
        0.,
        2. * 1.2 * spacing,
    );
    let mut nnps = LinkedNNPS::new(2, &world_bounds, dim);

    // solver data
    let dt = 1e-4;
    let mut t = 0.;
    let tf = 0.5;
    let mut step_no = 0;
    let pfreq = 100;

    let project_root = env!("CARGO_MANIFEST_DIR");
    let dir_name = project_root.to_owned() + "/dam_break_2d_output";
    let _p = fs::create_dir(&dir_name);

    // create a progress bar
    let total_steps = (tf / dt) as u64;
    let pb = setup_progress_bar(total_steps);

    // create kernel
    let kernel = CubicKernel::new(dim).expect("Something went wrong while creating Kernel");

    while t < tf {
        // stash the particles into the world's cells
        stash_2d(vec![&fluid, &tank], &mut nnps);

        // https://cg.informatik.uni-freiburg.de/publications/2014_EG_SPH_STAR.pdf
        // ----------------------
        // Algorithm 1
        // ----------------------
        fluid.rk2_initialize();
        tank.rk2_initialize();

        //////////////
        // stage 1  //
        //////////////


        reset_wcsph_entity(&mut fluid);
        reset_wcsph_entity(&mut tank);

        tait_equation(&mut fluid.p, &mut fluid.cs, &fluid.rho, 1000., 7., c0);
        tait_equation(&mut tank.p, &mut tank.cs, &tank.rho, 1000., 7., c0);

        continuity_and_momentum_eq_macro!(fluid, tank, nnps, kernel, 0.0, 0.0);
        continuity_and_momentum_eq_macro!(fluid, fluid, nnps, kernel, 0.0, 0.0);

        continuity_eq_macro!(tank, fluid, nnps, kernel);
        continuity_eq_macro!(tank, tank, nnps, kernel);

        wcsph::equations::apply_gravity(
            &mut fluid.au, &mut fluid.av, &mut fluid.aw,
            0.0, -9.81, 0.0,
        );

        fluid.rk2_stage_1(dt);
        tank.rk2_stage_1(dt);

        //////////////
        // stage 2  //
        //////////////

        reset_wcsph_entity(&mut fluid);
        reset_wcsph_entity(&mut tank);

        tait_equation(&mut fluid.p, &mut fluid.cs, &fluid.rho, 1000., 7., c0);
        tait_equation(&mut tank.p, &mut tank.cs, &tank.rho, 1000., 7., c0);

        continuity_and_momentum_eq_macro!(fluid, tank, nnps, kernel, 0.0, 0.0);
        continuity_and_momentum_eq_macro!(fluid, fluid, nnps, kernel, 0.0, 0.0);

        continuity_eq_macro!(tank, fluid, nnps, kernel);
        continuity_eq_macro!(tank, tank, nnps, kernel);

        wcsph::equations::apply_gravity(
            &mut fluid.au, &mut fluid.av, &mut fluid.aw,
            0.0, -9.81, 0.0,
        );

        fluid.rk2_stage_2(dt);
        tank.rk2_stage_2(dt);

        if step_no % pfreq == 0 {
            tank.write_vtk(format!("{}/tank_{}.vtk", &dir_name, step_no));
            fluid.write_vtk(format!("{}/fluid_{}.vtk", &dir_name, step_no));
        }
        step_no += 1;
        t += dt;

        // progress bar increment
        pb.inc(1);
    }
    pb.finish_with_message("Simulation succesfully completed");
}

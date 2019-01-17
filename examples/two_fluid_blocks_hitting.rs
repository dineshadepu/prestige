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

use prestige::geometry::grid_arange;

// std imports
use std::fs;

fn create_entites(spacing: f32) -> (WCSPH, WCSPH) {
    // Create two fluid blocks
    let (xf1, yf1) = grid_arange(1.0, 2.0, spacing , 0.0, 1.0, spacing);
    let mut fluid1 = WCSPH::new_with_xy(xf1.clone(), yf1, 0);
    let mf1 = 1000. * spacing.powf(2.);
    fluid1.rho = vec![1000.; xf1.len()];
    fluid1.m = vec![mf1; xf1.len()];
    fluid1.h = vec![1.2 * spacing; xf1.len()];
    fluid1.u = vec![-1.0; xf1.len()];

    let (xf2, yf2) = grid_arange(-0.1, 0.9, spacing, 0.0, 1.0, spacing);
    let mut fluid2 = WCSPH::new_with_xy(xf2.clone(), yf2, 1);
    let mf2 = 1000. * spacing.powf(2.);
    fluid2.rho = vec![1000.; xf2.len()];
    fluid2.m = vec![mf2; xf2.len()];
    fluid2.h = vec![1.2 * spacing; xf2.len()];
    fluid2.u = vec![1.0; xf2.len()];

    (fluid1, fluid2)
}

fn main() {
    let c0 = 1.;
    let spacing = 0.04;
    let dim = 2;

    let (mut fluid1, mut fluid2) = create_entites(spacing);

    print_no_part(vec![&fluid1.x, &fluid2.x]);

    // setup nnps
    let world_bounds = WorldBounds::new(
        -3.,
        3.,
        -5.,
        5.,
        0.,
        0.,
        2. * 1.2 * spacing,
    );
    let mut nnps = LinkedNNPS::new(2, &world_bounds, dim);

    // solver data
    let dt = 1e-4;
    let mut t = 0.;
    let tf = 1.;
    let mut step_no = 0;
    let pfreq = 100;

    let project_root = env!("CARGO_MANIFEST_DIR");
    let dir_name = project_root.to_owned() + "/two_fluid_blocks_hitting_output";
    let _p = fs::create_dir(&dir_name);

    // create a progress bar
    let total_steps = (tf / dt) as u64;
    let pb = setup_progress_bar(total_steps);

    // create kernel
    let kernel = CubicKernel::new(dim).expect("Something went wrong while creating Kernel");

    while t < tf {
        // stash the particles into the world's cells
        stash_2d(vec![&fluid1, &fluid2], &mut nnps);

        // https://cg.informatik.uni-freiburg.de/publications/2014_EG_SPH_STAR.pdf
        // ----------------------
        // Algorithm 1
        // ----------------------
        fluid1.rk2_initialize();
        fluid2.rk2_initialize();

        //////////////
        // stage 1  //
        //////////////


        reset_wcsph_entity(&mut fluid1);
        reset_wcsph_entity(&mut fluid2);

        tait_equation(&mut fluid1.p, &mut fluid1.cs, &fluid1.rho, 1000., 7., c0);
        tait_equation(&mut fluid2.p, &mut fluid2.cs, &fluid2.rho, 1000., 7., c0);

        continuity_and_momentum_eq_macro!(fluid1, fluid2, nnps, kernel, 0.0, 0.0);
        continuity_and_momentum_eq_macro!(fluid1, fluid1, nnps, kernel, 0.0, 0.0);

        continuity_and_momentum_eq_macro!(fluid2, fluid2, nnps, kernel, 0.0, 0.0);
        continuity_and_momentum_eq_macro!(fluid2, fluid1, nnps, kernel, 0.0, 0.0);

        // wcsph::equations::apply_gravity(
        //     &mut fluid1.au, &mut fluid1.av, &mut fluid1.aw,
        //     0.0, -9.81, 0.0,
        // );

        fluid1.rk2_stage_1(dt);
        fluid2.rk2_stage_1(dt);

        //////////////
        // stage 2  //
        //////////////

        reset_wcsph_entity(&mut fluid1);
        reset_wcsph_entity(&mut fluid2);

        tait_equation(&mut fluid1.p, &mut fluid1.cs, &fluid1.rho, 1000., 7., c0);
        tait_equation(&mut fluid2.p, &mut fluid2.cs, &fluid2.rho, 1000., 7., c0);

        continuity_and_momentum_eq_macro!(fluid1, fluid2, nnps, kernel, 0.0, 0.0);
        continuity_and_momentum_eq_macro!(fluid1, fluid1, nnps, kernel, 0.0, 0.0);

        continuity_and_momentum_eq_macro!(fluid2, fluid2, nnps, kernel, 0.0, 0.0);
        continuity_and_momentum_eq_macro!(fluid2, fluid1, nnps, kernel, 0.0, 0.0);

        // wcsph::equations::apply_gravity(
        //     &mut fluid1.au, &mut fluid1.av, &mut fluid1.aw,
        //     0.0, -9.81, 0.0,
        // );

        fluid1.rk2_stage_2(dt);
        fluid2.rk2_stage_2(dt);

        if step_no % pfreq == 0 {
            fluid2.write_vtk(format!("{}/fluid2_{}.vtk", &dir_name, step_no));
            fluid1.write_vtk(format!("{}/fluid1_{}.vtk", &dir_name, step_no));
        }
        step_no += 1;
        t += dt;

        // progress bar increment
        pb.inc(1);
    }
    pb.finish_with_message("Simulation succesfully completed");
}

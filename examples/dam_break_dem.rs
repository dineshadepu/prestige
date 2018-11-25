extern crate prestige;
extern crate simple_shapes;

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
use std::fs;

fn main() {
    let spacing = 0.1;
    let (xt, yt) = tank(-1., 3., spacing, -1., 4., spacing, 2);
    let (xb, yb) = grid_arange(
        1.5,
        2.5 + spacing / 2.,
        spacing,
        1.5,
        2.5 + spacing / 2.,
        spacing,
    );
    let body_particle_no = xb.len();
    let tank_particle_no = xt.len();
    let mut body = DEM {
        x: xb.clone(),
        y: yb.clone(),
        x0: xb.clone(),
        y0: yb.clone(),
        u: vec![0.; xb.len()],
        v: vec![0.; xb.len()],
        u0: vec![0.; xb.len()],
        v0: vec![0.; xb.len()],
        r: vec![spacing / 2.; body_particle_no],
        fx: vec![0.; body_particle_no],
        fy: vec![0.; body_particle_no],
        h: vec![spacing / 2.; body_particle_no],
        nnps_idx: 0,
        no_par: body_particle_no,
        m: vec![1.; body_particle_no],
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
        m: vec![1.; tank_particle_no],
    };

    println!(
        "Body particles: {}, tank particles: {}, Total particles: {}",
        body.x.len(),
        tank.x.len(),
        body.x.len() + tank.x.len()
    );

    // solver data
    let dt = 1e-4;
    let mut tf = 100. * dt;
    let mut step_no = 0;
    let pfreq = 10;

    let version = env!("CARGO_MANIFEST_DIR");
    let dir_name = version.to_owned() + "/dam_break_dem_output";
    fs::create_dir(&dir_name);

    while tf > 0. {
        // stash the particles into the world's cells
        let nnps = stash(vec![&body, &tank]);

        body.initialize();
        body_force(&mut body.fx, &mut body.fy, 0.0, -9.81);

        contact_force_par(
            &body.x,
            &body.y,
            &mut body.fx,
            &mut body.fy,
            &tank.x,
            &tank.y,
            tank.nnps_idx,
            &nnps,
        );

        body.stage1(dt);

        contact_force_par(
            &body.x,
            &body.y,
            &mut body.fx,
            &mut body.fy,
            &tank.x,
            &tank.y,
            tank.nnps_idx,
            &nnps,
        );

        body.stage1(dt);

        if step_no % pfreq == 0 {
            // println!("{}", step_no);
            let filename = format!("{}/tank_{}.vtk", &dir_name, step_no);
            tank.write_vtk(filename);

            let filename = format!("{}/body_{}.vtk", &dir_name, step_no);
            body.write_vtk(filename);
            println!("{} ", step_no);
        }
        step_no += 1;
        tf -= dt;
    }
}

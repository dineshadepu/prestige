extern crate prestige;
extern crate simple_shapes;

// crates imports
use prestige::{
    contact_search::{get_neighbours_2d, stash_2d, WorldBounds, NNPS},
    physics::dem::{equations::body_force, DEM},
    RK2Integrator,
};
use simple_shapes::{grid_arange, tank};

// std imports
use std::env;

// define serial force equation
pub fn contact_force(
    d_x: &[f32],
    d_y: &[f32],
    d_r: &[f32],
    d_fx: &mut [f32],
    d_fy: &mut [f32],
    s_x: &[f32],
    s_y: &[f32],
    s_r: &[f32],
    s_nnps_id: usize,
    nnps: &NNPS,
    kn: f32,
) {
    for i in 0..d_x.len() {
        let nbrs = get_neighbours_2d(d_x[i], d_y[i], s_nnps_id, &nnps);
        for &j in nbrs.iter() {
            let dx = d_x[i] - s_x[j];
            let dy = d_y[i] - s_y[j];
            let rij = (dx.powf(2.) + dy.powf(2.)).sqrt();
            // eliminate the interaction between same particle
            if rij > 1e-12 {
                // if the two particles are in overlap
                let nij_x = dx / rij;
                let nij_y = dy / rij;

                let overlap = d_r[i] + s_r[j] - rij;
                if overlap > 0. {
                    d_fx[i] += kn * overlap * nij_x;
                    d_fy[i] += kn * overlap * nij_y;
                }
            }
        }
    }
}

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
    tank.r = vec![spacing / 4.; tank_particle_no];
    tank.nnps_idx = 1;

    let kn = 1e7;
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

    while t < tf {
        // stash the particles into the world's cells
        stash_2d(vec![&body, &tank], &mut nnps);

        body.initialize();

        // stage 2
        body_force(&mut body.fx, &mut body.fy, &body.m, 0.0, -9.81);
        contact_force(
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
        contact_force(
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
        contact_force(
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
        contact_force(
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

        t += dt;
    }
}

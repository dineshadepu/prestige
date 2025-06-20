use prestige::*;

fn main() {

    let dim = 3;

    let x_sand = vec![1.0, 2.0];
    let y_sand = vec![1.0, 1.0];
    let z_sand = vec![0.0, 0.0];
    let neighbour_radius_sand = vec![0.2, 0.2];
    let rho_sand = vec![1000.0, 1000.0];
    let radius_sand = vec![0.1, 0.1];
    let dem_id_sand = vec![0, 0];

    let mut sand = ParticlesDEM::new(dim, "sand".to_string(), &x_sand, &y_sand, &z_sand, &neighbour_radius_sand, &rho_sand, &radius_sand, &dem_id_sand);

    let x_body = vec![1.0, 2.0];
    let y_body = vec![-0.5, 0.5];
    let z_body = vec![0.0, 0.0];
    let neighbour_radius_body = vec![0.2, 0.2];
    let rho_body = vec![1000.0, 1000.0];
    let radius_body = vec![0.1, 0.1];
    let dem_id_body = vec![1, 1];
    let mut body = ParticlesDEM::new(dim, "body".to_string(), &x_body, &y_body, &z_body, &neighbour_radius_body, &rho_body, &radius_body, &dem_id_body);

    let x_boundary = vec![1.0, 2.0];
    let y_boundary = vec![-0.5, 0.5];
    let z_boundary = vec![0.0, 0.0];
    let neighbour_radius_boundary = vec![0.2, 0.2];
    let rho_boundary = vec![1000.0, 1000.0];
    let radius_boundary = vec![0.1, 0.1];
    let dem_id_boundary = vec![2, 2];
    let mut boundary = ParticlesDEM::new(dim, "boundary".to_string(), &x_boundary, &y_boundary, &z_boundary, &neighbour_radius_boundary, &rho_boundary, &radius_boundary, &dem_id_boundary);

    let dt = 1e-4;
    let tf = 1.;
    let num_steps = (tf / dt) as usize;
    let mut t = 0.;
    for step in 0..=num_steps {
        // loop body
        sand.stage1(dt);

        sand.stage2(dt);

        // update the neighbours
        // neighbors->build( x, 0, x.size(), delta,
        //                   cell_ratio, mesh_min, mesh_max );

        // apply_hertz_contact_force_DEM_macro!(sand, (sand, body, boundary), t, dt, 0.2, 0.3);

        // println!("Updated fx: {:?}", sand.get_fx());


        sand.stage3(dt);

        // sand.output_particles( step );
        sand.output_data(t);
        t = t + dt;
    }
}

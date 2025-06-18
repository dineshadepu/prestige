use prestige::*;

fn main() {

    let dim = 3;

    let x_sand = vec![1.0, 2.0];
    let y_sand = vec![0.0, 0.0];
    let z_sand = vec![0.0, 0.0];
    let rho_sand = vec![1000.0, 1000.0];
    let radius_sand = vec![0.1, 0.1];

    let mut sand = ParticlesDEM::new(dim, "sand".to_string(), &x_sand, &y_sand, &z_sand, &rho_sand, &radius_sand);

    let x_body = vec![1.0, 2.0];
    let y_body = vec![-0.5, 0.5];
    let z_body = vec![0.0, 0.0];
    let rho_body = vec![1000.0, 1000.0];
    let radius_body = vec![0.1, 0.1];
    let mut body = ParticlesDEM::new(dim, "body".to_string(), &x_body, &y_body, &z_body, &rho_body, &radius_body);

    let x_boundary = vec![1.0, 2.0];
    let y_boundary = vec![-0.5, 0.5];
    let z_boundary = vec![0.0, 0.0];
    let rho_boundary = vec![1000.0, 1000.0];
    let radius_boundary = vec![0.1, 0.1];
    let mut boundary = ParticlesDEM::new(dim, "boundary".to_string(), &x_boundary, &y_boundary, &z_boundary, &rho_boundary, &radius_boundary);

    // unsafe{
    //     let ptr = &mut sand as *mut _;
    //     apply_force(ptr, ptr as *const _);
    // }
    apply_force([&mut sand, &mut body, &mut boundary], true);

    // println!("Updated fx: {:?}", particles.get_fx());
}

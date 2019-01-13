use super::test_collision_detection::create_a_simple_2d_test_entity;
use super::brute_force_nbrs;

#[test]
fn test_uniform_grid() {
    let e = create_a_simple_2d_test_entity(0.04);
    let brute_nbrs = brute_force_nbrs((e.x[3], e.y[3], e.z[3]), 0.1, &e);
    println!("neighbours are {:?}", brute_nbrs);
}

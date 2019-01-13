#[macro_use]
extern crate criterion;
extern crate prestige;

use criterion::{Criterion, Fun, Bencher};
use prestige::contact_search::{
    brute_force_nbrs,
    linked_nnps::{stash_2d, LinkedNNPS, WorldBounds},
    test_collision_detection::create_a_simple_2d_test_entity,
    GetXYZH, NNPSGeneric,
};

fn bench_get_neighbours_brute_and_llnnps(c: &mut Criterion) {
    // create simple entity
    let spacing = 0.04;
    let radius = 0.1;
    let e = create_a_simple_2d_test_entity(spacing);
    // create ll nnps object
    let dim = 2;
    let no_entites = 1;
    let (x_min, x_max, y_min, y_max, z_min, z_max) = (0.0, 1.0, 0.0, 1.0, 0.0, 0.0);
    let world_bound = WorldBounds::new(x_min, x_max, y_min, y_max, z_min, z_max, radius);
    let mut nnps = LinkedNNPS::new(no_entites, &world_bound, dim);

    // stash the particles
    stash_2d(vec![&e], &mut nnps);
    let nbrs = nnps.get_neighbours(e.x[3], e.y[3], 0., 0);

    let brute = Fun::new("brute", move |b, _i| {
        b.iter(|| brute_force_nbrs((e.x[3], e.y[3], e.z[3]), 0.1, &e))
    });

    let e = create_a_simple_2d_test_entity(spacing);
    let llnnps = Fun::new("llnnps", move |b, _i| {
        b.iter(|| nnps.get_neighbours(e.x[3], e.y[3], e.z[3], 0))
    });

    let functions = vec!(brute, llnnps);

    c.bench_functions("Get Neighbours", functions, &1.);
}

criterion_group!(benches, bench_get_neighbours_brute_and_llnnps);
criterion_main!(benches);

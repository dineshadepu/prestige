use super::{
    super::kernel::CubicKernel,
    equations::{continuity_equation, reset_wcsph_entity},
    WCSPH,
};

use crate::{
    contact_search::{
        linked_nnps::{stash_2d, LinkedNNPS, WorldBounds},
        test_collision_detection::test_nbrs,
        NNPSGeneric,
    },
    continuity_eq_macro,
    geometry::grid_arange,
};

#[test]
fn test_continuity_equation() {
    // create two entities approaching each other
    /////////////////
    // test case 1 //
    /////////////////
    let dim = 2;
    let spacing = 0.1;
    let radius = 2. * 1.2 * spacing;
    let max_size = radius;
    let (x1, y1) = grid_arange(-1., 0., spacing, 0., 1.0, spacing);
    let (x2, y2) = grid_arange(1., 2., spacing, 0., 1.0, spacing);
    let mut block1 = WCSPH::new_with_xy(x1.clone(), y1, 0);
    let mc = 1000. * spacing.powf(2.);
    block1.rho = vec![1000.; x1.len()];
    block1.m = vec![mc; x1.len()];
    block1.h = vec![1.2 * spacing; x1.len()];
    let mut block2 = WCSPH::new_with_xy(x2.clone(), y2, 1);
    let rho_2 = 1000.;
    let m2 = rho_2 * spacing.powf(3.);
    block2.m = vec![m2; x2.len()];
    block2.h = vec![1.2 * spacing; x2.len()];
    block2.rho = vec![1000.; x2.len()];

    // setup nnps
    let world_bounds = WorldBounds::new(-2., 2., -2., 2., 0., 0., max_size);
    let kernel = CubicKernel::new(dim).expect("Something went wrong while creating Kernel");

    let mut nnps = LinkedNNPS::new(2, &world_bounds, dim);
    stash_2d(vec![&block1, &block2], &mut nnps);

    // before executing any equations check if the neighbours are tested
    // select a random particle, index (i)
    // uncomment to test neighbours
    // test_nbrs(vec![&block1, &block2], &nnps, radius);

    reset_wcsph_entity(&mut block1); // this will reset accelerations of density, velocity
    reset_wcsph_entity(&mut block2); // this will reset accelerations of density, velocity

    // Influence of block 2 on block 1
    continuity_eq_macro!(block1, block2, nnps, kernel);
    // since block 1 and block 2 are very far away and they wont interact
    // because no particle in block 2 is in the scaled smoothing radius of
    // particle 1

    // Which should result in zero acceleration in arho, let's check that
    assert_eq!(block1.arho, vec![0.; block1.x.len()]);

    // Influence of block 1 on itself
    continuity_eq_macro!(block1, block1, nnps, kernel);

    // even this should not generate any change in the acceleration
    // because there is no relative velocity between the particles
    assert_eq!(block1.arho, vec![0.; block1.x.len()]);

    /////////////////
    // test case 2 //
    /////////////////
    // This case is similar to the previous one but now the particles will
    // get together
    let (x1, y1) = grid_arange(1.15, 2.15, spacing, 0., 1.0, spacing);
    block1.x = x1;
    block1.y = y1;
    // restash the particles
    stash_2d(vec![&block1, &block2], &mut nnps);

    // before executing any equations check if the neighbours are tested
    // select a random particle, index (i)
    // uncomment to test neighbours
    // test_nbrs(vec![&block1, &block2], &nnps, radius);

    // Influence of block 2 on block 1
    continuity_eq_macro!(block1, block2, nnps, kernel);
    // Even though block 1 and block 2 are close, but the velocities are
    // zero, there wont be any acceleration in density
    assert_eq!(block1.arho, vec![0.; block1.x.len()]);

    /////////////////
    // test case 3 //
    /////////////////
    // let's give some velocity to block 2. This will create relative velocity
    // among the particles of block 1 and block 2 and generate acceleration to
    // both block 1 and block 2. This generation is only due to external source
    // which means block 1 acceleration is due to block 2 and block 2
    // acceleration is due to block 1, but there wont be any self induction, as
    // we will see in next few steps.
    block2.u = vec![1.; block2.x.len()];

    // self influence will not lead to any acceleration
    continuity_eq_macro!(block1, block1, nnps, kernel);
    assert_eq!(block1.arho, vec![0.; block1.x.len()]);
    continuity_eq_macro!(block2, block2, nnps, kernel);
    assert_eq!(block2.arho, vec![0.; block2.x.len()]);

    // but other there is a cross influence
    continuity_eq_macro!(block1, block2, nnps, kernel);
    println!("{:?}", block1.arho );
    for arho_i in &block1.arho {
        assert!(*arho_i > 0.)
    }
    continuity_eq_macro!(block2, block1, nnps, kernel);
    for arho_i in &block2.arho {
        assert!(*arho_i > 0.)
    }
}

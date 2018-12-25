use prestige::{
    contact_search::{
        get_neighbours_2d, get_neighbours_3d, stash_2d, stash_3d, GetXYZH, WorldBounds, NNPS,
    },
    impl_GetXYZH,
};
use rand::Rng;
use simple_shapes::{grid_arange, grid_arange_3d};

/// A simple struct used for nnps tests
/// It has x, y, z and h as it's attributes and will derive a
/// GETXYZH trait.
struct Simple {
    pub x: Vec<f32>,
    pub y: Vec<f32>,
    pub z: Vec<f32>,
    pub nnps_idx: usize,
    pub h: Vec<f32>,
}

impl Simple {
    fn new_fromxy(x: Vec<f32>, y: Vec<f32>, nnps_idx: usize) -> Self {
        let no_par = x.len();
        Simple {
            x,
            y,
            z: vec![0.; no_par],
            nnps_idx,
            h: vec![0.; no_par],
        }
    }
}
impl_GetXYZH!(Simple);

#[test]
fn test_nnps_attributes_1d() {
    // -------------------------------------
    // Test case 1
    // given bounds of the world calculate the number of cells in
    // nnps object
    let dim = 2;
    let no_entites = 1;
    let (x_min, x_max, y_min, y_max, z_min, z_max, max_size) = (0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.1);
    let world_bound = WorldBounds::new(x_min, x_max, y_min, y_max, z_min, z_max, max_size);
    let nnps = NNPS::new(no_entites, &world_bound, dim);

    // check number of cells in each direction
    // with a length of 1.0 meter in each direction and max size of the particle
    // as 0.1, we expect the number of cells end and starting points to look like
    //
    //
    //
    //
    // |-------|-------|------|------|------|------|------|------|------|------|------|
    // 0.0     0.1    0.2    0.3    0.4    0.5    0.6    0.7    0.8    0.9    1.0    1.1
    //
    //
    //
    // As we can see the number of cells in each direction are 11
    // to get all the particles in the domain we need make sure that the end point also
    // gets in the cells list,
    // because there could be points at position like 1.01, to include the points
    // of suck kind, while partitioning we will add one cell size to out domain
    // and create cells

    assert_eq!(11, nnps.no_x_cells);
    assert_eq!(11, nnps.no_y_cells);
    // in 2-d z direction has only one (plane of) cell(s)
    assert_eq!(1, nnps.no_z_cells);
    // check number of cells (total cells)
    assert_eq!(nnps.cells.len(), 121);
    // check the sub cells of cell. In the present case we have one entity
    // so the length of the cell will be 1
    assert_eq!(nnps.cells[0].indices.len(), 1);

    // -------------------------------------
    // Test case 2
    // given bounds of the world calculate the number of cells in
    // nnps object
    let dim = 2;
    let no_entites = 3;
    let (x_min, x_max, y_min, y_max, z_min, z_max, max_size) = (0.0, 1.05, 0.0, 1.0, 0.0, 0.0, 0.1);
    let world_bound = WorldBounds::new(x_min, x_max, y_min, y_max, z_min, z_max, max_size);
    let nnps = NNPS::new(no_entites, &world_bound, dim);

    //
    //
    // |-------|-------|------|------|------|------|------|------|------|------|   .
    // 0.0     0.1    0.2    0.3    0.4    0.5    0.6    0.7    0.8    0.9    1.0  1.05
    //
    //
    // Difference between the previous example and this one is we have a point
    // at 1.05, now we need to check, if the cells are 11, because we should also incorporate
    // all the points

    assert_eq!(11, nnps.no_x_cells);
    assert_eq!(11, nnps.no_y_cells);
    // in 2-d z direction has only one (plane of) cell(s)
    assert_eq!(1, nnps.no_z_cells);
    assert_eq!(nnps.cells.len(), 121);
    // check the sub cells of cell. In the present case we have 3 entities
    // so the length of the cell will be 3
    assert_eq!(nnps.cells[0].indices.len(), 3);
    // -------------------------------------
    // Test case 3
    // given bounds of the world calculate the number of cells in
    // nnps object
    let dim = 3;
    let no_entites = 3;
    let (x_min, x_max, y_min, y_max, z_min, z_max, max_size) =
        (0.0, 1.05, 0.0, 1.0, 0.0, 0.22, 0.1);
    let world_bound = WorldBounds::new(x_min, x_max, y_min, y_max, z_min, z_max, max_size);
    let nnps = NNPS::new(no_entites, &world_bound, dim);

    assert_eq!(11, nnps.no_x_cells);
    assert_eq!(11, nnps.no_y_cells);
    // in 2-d z direction has only one (plane of) cell(s)
    assert_eq!(3, nnps.no_z_cells);
    assert_eq!(nnps.cells.len(), 363);
    // check the sub cells of cell. In the present case we have 3 entities
    // so the length of the cell will be 3
    assert_eq!(nnps.cells[0].indices.len(), 3);

    // ----------------------------------------
    // Test case 4
    // Set the dimension to be 2, but give z limits. This should create only one z_direction
    // cell
    let dim = 2;
    let no_entites = 1;
    let (x_min, x_max, y_min, y_max, z_min, z_max, max_size) =
        (0.0, 1.05, 0.0, 1.0, 0.0, 0.22, 0.1);
    let world_bound = WorldBounds::new(x_min, x_max, y_min, y_max, z_min, z_max, max_size);
    let nnps = NNPS::new(no_entites, &world_bound, dim);

    assert_eq!(11, nnps.no_x_cells);
    assert_eq!(11, nnps.no_y_cells);
    // in 2-d z direction has only one (plane of) cell(s)
    assert_eq!(1, nnps.no_z_cells);
    assert_eq!(nnps.cells.len(), 121);
    // check the sub cells of cell. In the present case we have 3 entities
    // so the length of the cell will be 3
    assert_eq!(nnps.cells[0].indices.len(), 1);
}

#[test]
fn test_stash_2d_single_entity() {
    // About the test:
    // Given all particles, check if the particles are stashed properly
    // into their respective cells

    // create nnps with 5 `x_cells` and 8 `y_cells`, with a total
    // 40 cells.
    let dim = 2;
    let no_entites = 1;
    let (x_min, x_max, y_min, y_max, z_min, z_max, max_size) = (0.0, 0.4, 0.0, 0.7, 0.0, 0.0, 0.1);
    let world_bound = WorldBounds::new(x_min, x_max, y_min, y_max, z_min, z_max, max_size);
    let mut nnps = NNPS::new(no_entites, &world_bound, dim);
    assert_eq!(5, nnps.no_x_cells);
    assert_eq!(8, nnps.no_y_cells);
    assert_eq!(1, nnps.no_z_cells);
    assert_eq!(nnps.cells.len(), 40);
    assert_eq!(nnps.cells[0].indices.len(), 1);

    // ----------------------------------------------
    // Test case 1
    // create entity
    let x0 = vec![0.05, 0.15, 0.25, 0.51, -0.1];
    let y0 = vec![0.05; 5];
    let nnps_id0 = 0;
    let smpl_0 = Simple::new_fromxy(x0, y0, nnps_id0);

    // stash the particles
    stash_2d(vec![&smpl_0], &mut nnps);

    // since the domain of nnps in x direction is from 0.0 to 0.4, and nnps
    // extends it to 0.5, the fourth point is out of domain, so it wont be
    // stashed in our nnps. The point which is left to nnps minimum (-0.1) will
    // also not be stashed.
    assert_eq!(true, nnps.cells[0].indices[nnps_id0].contains(&0));
    assert_eq!(true, nnps.cells[1].indices[nnps_id0].contains(&1));
    assert_eq!(true, nnps.cells[2].indices[nnps_id0].contains(&2));

    // ----------------------------------------------
    // Test case 2
    // write in a paper to visualize how this looks
    // Note that the last two particles are not in the nnps domain.
    let x0 = vec![
        0.05, 0.15, 0.25, 0.05, 0.15, 0.25, 0.05, 0.15, 0.25, -110., 110.,
    ];
    let y0 = vec![
        0.05, 0.05, 0.05, 0.15, 0.15, 0.15, 0.25, 0.25, 0.25, 10.25, -30.25,
    ];
    let nnps_id0 = 0;
    let smpl_0 = Simple::new_fromxy(x0.clone(), y0.clone(), nnps_id0);
    stash_2d(vec![&smpl_0], &mut nnps);

    // println!("{:?}", nnps);
    // check all the points except the last 2
    for i in 0..x0.len() - 2 {
        let x_cell_idx = ((x0[i] - x_min) / max_size) as usize;
        let y_cell_idx = ((y0[i] - y_min) / max_size) as usize;
        let cell_idx = x_cell_idx + y_cell_idx * nnps.no_x_cells;
        assert_eq!(true, nnps.cells[cell_idx].indices[nnps_id0].contains(&i));
        assert_eq!(1, nnps.cells[cell_idx].indices[nnps_id0].len());
    }

    // test the last two points explicitly. They shouldn't be stashed in any of the cells
    for i in 0..nnps.cells.len() {
        assert_eq!(false, nnps.cells[i].indices[nnps_id0].contains(&9));
        assert_eq!(false, nnps.cells[i].indices[nnps_id0].contains(&10));
    }
}

#[test]
fn test_stash_2d_multiple_entity() {
    // About the test:
    // Given all particles with many entities, check if the particles are stashed properly
    // into their respective cells

    // create nnps with 5 `x_cells` and 8 `y_cells`, with a total
    // 40 cells.
    let dim = 2;
    let no_entites = 3;
    let (x_min, x_max, y_min, y_max, z_min, z_max, max_size) = (0.0, 0.4, 0.0, 0.7, 0.0, 0.0, 0.1);
    let world_bound = WorldBounds::new(x_min, x_max, y_min, y_max, z_min, z_max, max_size);
    let mut nnps = NNPS::new(no_entites, &world_bound, dim);
    assert_eq!(5, nnps.no_x_cells);
    assert_eq!(8, nnps.no_y_cells);
    assert_eq!(1, nnps.no_z_cells);
    assert_eq!(nnps.cells.len(), 40);
    assert_eq!(nnps.cells[0].indices.len(), 3);

    // ----------------------------------------------
    // Test case 1
    // create entities
    let x0 = vec![0.05, 0.15, 0.25, 0.51, -0.1];
    let y0 = vec![0.05; 5];
    let nnps_id0 = 0;
    let smpl_0 = Simple::new_fromxy(x0.clone(), y0.clone(), nnps_id0);
    let nnps_id1 = 1;
    let smpl_1 = Simple::new_fromxy(x0.clone(), y0.clone(), nnps_id1);
    let nnps_id2 = 2;
    let smpl_2 = Simple::new_fromxy(x0.clone(), y0.clone(), nnps_id2);

    // stash the particles
    stash_2d(vec![&smpl_0, &smpl_1, &smpl_2], &mut nnps);

    // since the domain of nnps in x direction is from 0.0 to 0.4, and nnps
    // extends it to 0.5, the fourth point is out of domain, so it wont be
    // stashed in our nnps. The point which is left to nnps minimum (-0.1) will
    // also not be stashed.
    for nnps_id in 0..no_entites {
        assert_eq!(true, nnps.cells[0].indices[nnps_id].contains(&0));
        assert_eq!(true, nnps.cells[1].indices[nnps_id].contains(&1));
        assert_eq!(true, nnps.cells[2].indices[nnps_id].contains(&2));
    }

    // ----------------------------------------------
    // Test case 2
    // write in a paper to visualize how this looks
    // Note that the last two particles are not in the nnps domain.
    // Same as previous test case, but here we have many entities
    let x0 = vec![
        0.05, 0.15, 0.25, 0.05, 0.15, 0.25, 0.05, 0.15, 0.25, -110., 110.,
    ];
    let y0 = vec![
        0.05, 0.05, 0.05, 0.15, 0.15, 0.15, 0.25, 0.25, 0.25, 10.25, -30.25,
    ];
    let nnps_id0 = 0;
    let smpl_0 = Simple::new_fromxy(x0.clone(), y0.clone(), nnps_id0);
    let x1 = vec![
        0.05, 0.15, 0.25, 0.05, 0.15, 0.25, 0.05, 0.15, 0.25, -110., 110.,
    ];
    let y1 = vec![
        0.05, 0.05, 0.05, 0.15, 0.15, 0.15, 0.25, 0.25, 0.25, 10.25, -30.25,
    ];
    let nnps_id1 = 1;
    let smpl_1 = Simple::new_fromxy(x1.clone(), y1.clone(), nnps_id1);
    let x2 = vec![
        0.05, 0.15, 0.25, 0.05, 0.15, 0.25, 0.05, 0.15, 0.25, -110., 110.,
    ];
    let y2 = vec![
        0.05, 0.05, 0.05, 0.15, 0.15, 0.15, 0.25, 0.25, 0.25, 10.25, -30.25,
    ];
    let nnps_id2 = 2;
    let smpl_2 = Simple::new_fromxy(x2.clone(), y2.clone(), nnps_id2);

    stash_2d(vec![&smpl_0, &smpl_1, &smpl_2], &mut nnps);

    // check all the points except the last 2
    for nnps_id in 0..no_entites {
        for i in 0..x0.len() - 2 {
            let x_cell_idx = ((x0[i] - x_min) / max_size) as usize;
            let y_cell_idx = ((y0[i] - y_min) / max_size) as usize;
            let cell_idx = x_cell_idx + y_cell_idx * nnps.no_x_cells;
            assert_eq!(true, nnps.cells[cell_idx].indices[nnps_id].contains(&i));
            assert_eq!(1, nnps.cells[cell_idx].indices[nnps_id].len());
        }
    }

    // test the last two points explicitly. They shouldn't be stashed in any of the cells
    for i in 0..nnps.cells.len() {
        for nnps_id in 0..no_entites {
            assert_eq!(false, nnps.cells[i].indices[nnps_id].contains(&9));
            assert_eq!(false, nnps.cells[i].indices[nnps_id].contains(&10));
        }
    }
}

#[test]
fn test_get_neighbours_2d() {
    // ------------------------------------------
    // ------------------------------------------
    // Test case 1
    // create a 2d domain
    let spacing = 0.05;
    let (x, y) = grid_arange(0.0, 1.0, spacing, 0.0, 1.0, spacing);
    let smpl = Simple::new_fromxy(x.clone(), y.clone(), 0);
    let radius = 4. * spacing;
    let max_size = radius;

    // create nnps object
    let dim = 2;
    let no_entites = 1;
    let (x_min, x_max, y_min, y_max, z_min, z_max) = (0.0, 1.0, 0.0, 1.0, 0.0, 0.0);
    let world_bound = WorldBounds::new(x_min, x_max, y_min, y_max, z_min, z_max, max_size);
    let mut nnps = NNPS::new(no_entites, &world_bound, dim);

    // stash the particles
    stash_2d(vec![&smpl], &mut nnps);

    for i in (0..x.len()).step_by(5) {
        // get the neighbours from nnps
        let nbrs = get_neighbours_2d(x[i], y[i], 0., smpl.nnps_idx, &nnps);
        let mut filtered_nbrs = vec![];
        // select the neighbours which are in limit or radius scale.
        for &j in nbrs.iter() {
            let rij = ((x[i] - x[j]).powf(2.) + (y[i] - y[j]).powf(2.)).sqrt();
            if rij < radius {
                filtered_nbrs.push(j);
            }
        }
        // sort the filtered neighbours
        filtered_nbrs.sort();

        // compute neighbours by brute force
        let mut brute_nbrs = vec![];
        for j in 0..x.len() {
            let rij = ((x[i] - x[j]).powf(2.) + (y[i] - y[j]).powf(2.)).sqrt();
            if rij < radius {
                brute_nbrs.push(j);
            }
        }
        brute_nbrs.sort();

        assert_eq!(filtered_nbrs, brute_nbrs);
    }

    // ------------------------------------------
    // ------------------------------------------
    // Test case 2

    // helpers to create a random domain
    let mut rng = rand::thread_rng();
    let spacing = 0.05;
    let no_particles = 1000;
    let radius = 4. * spacing;
    let max_size = radius;

    // create nnps object
    let dim = 2;
    let no_entites = 1;
    let (x_min, x_max, y_min, y_max, z_min, z_max) = (0.0, 10.0, 0.0, 10.0, 0.0, 0.0);
    let world_bound = WorldBounds::new(x_min, x_max, y_min, y_max, z_min, z_max, max_size);
    let mut nnps = NNPS::new(no_entites, &world_bound, dim);

    // create a 2d domain but with random numbers

    let (mut x, mut y) = (vec![], vec![]);
    for _ in 0..no_particles {
        x.push(rng.gen_range(x_min, x_max));
        y.push(rng.gen_range(y_min, y_max));
    }
    x[0] = -100.;
    x[1] = 100.;
    y[2] = 100.;
    y[3] = 100.;
    x[4] = -100.;
    y[4] = -100.;
    let smpl = Simple::new_fromxy(x.clone(), y.clone(), 0);

    // stash the particles
    stash_2d(vec![&smpl], &mut nnps);

    for i in (0..x.len()).step_by(5) {
        // get the neighbours from nnps
        let nbrs = get_neighbours_2d(x[i], y[i], 0., smpl.nnps_idx, &nnps);
        // println!("Length of pure neighbours {:?}", nbrs.len());
        let mut filtered_nbrs = vec![];
        // select the neighbours which are in limit or radius scale.
        for &j in nbrs.iter() {
            let rij = ((x[i] - x[j]).powf(2.) + (y[i] - y[j]).powf(2.)).sqrt();
            if rij < radius {
                filtered_nbrs.push(j);
            }
        }
        // sort the filtered neighbours
        filtered_nbrs.sort();
        // println!("Length of filtered neighbours {:?}", filtered_nbrs.len());

        // compute neighbours by brute force
        let mut brute_nbrs = vec![];
        for j in 0..x.len() {
            let rij = ((x[i] - x[j]).powf(2.) + (y[i] - y[j]).powf(2.)).sqrt();
            if rij < radius {
                if (x[i] >= nnps.x_min && x[i] <= nnps.x_max)
                    && (y[i] >= nnps.y_min && y[i] <= nnps.y_max)
                {
                    brute_nbrs.push(j);
                }
            }
        }
        brute_nbrs.sort();
        // println!("-------------------------------",);

        assert_eq!(filtered_nbrs, brute_nbrs);
    }
}

#[test]
fn test_get_neighbours_3d() {
    // ------------------------------------------
    // ------------------------------------------
    // Test case 1
    // create a 3d domain
    let spacing = 0.1;
    let (x, y, z) = grid_arange_3d(0.0, 1.0, spacing, 0.0, 1.0, spacing, 0.0, 1.0, spacing);
    let mut smpl = Simple::new_fromxy(x.clone(), y.clone(), 0);
    smpl.z = z.clone();
    let radius = 4. * spacing;
    let max_size = radius;

    // create nnps object
    let dim = 3;
    let no_entites = 1;
    let (x_min, x_max, y_min, y_max, z_min, z_max) = (0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
    let world_bound = WorldBounds::new(x_min, x_max, y_min, y_max, z_min, z_max, max_size);
    let mut nnps = NNPS::new(no_entites, &world_bound, dim);
    // println!("{:?}", nnps);

    // stash the particles
    stash_3d(vec![&smpl], &mut nnps);

    for i in (0..x.len()).step_by(5) {
        // get the neighbours from nnps
        let nbrs = get_neighbours_3d(x[i], y[i], z[i], smpl.nnps_idx, &nnps);
        let mut filtered_nbrs = vec![];
        // select the neighbours which are in limit or radius scale.
        for &j in nbrs.iter() {
            let rij =
                ((x[i] - x[j]).powf(2.) + (y[i] - y[j]).powf(2.) + (z[i] - z[j]).powf(2.)).sqrt();
            if rij < radius {
                filtered_nbrs.push(j);
            }
        }
        // sort the filtered neighbours
        filtered_nbrs.sort();

        // compute neighbours by brute force
        let mut brute_nbrs = vec![];
        for j in 0..x.len() {
            let rij =
                ((x[i] - x[j]).powf(2.) + (y[i] - y[j]).powf(2.) + (z[i] - z[j]).powf(2.)).sqrt();
            if rij < radius {
                brute_nbrs.push(j);
            }
        }
        brute_nbrs.sort();

        assert_eq!(filtered_nbrs, brute_nbrs);
    }

    // ------------------------------------------
    // ------------------------------------------
    // Test case 2

    // helpers to create a random domain
    let mut rng = rand::thread_rng();
    let spacing = 0.05;
    let no_particles = 1000;
    let radius = 2. * spacing;
    let max_size = radius;

    // create nnps object
    let dim = 3;
    let no_entites = 1;
    let (x_min, x_max, y_min, y_max, z_min, z_max) = (0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
    let world_bound = WorldBounds::new(x_min, x_max, y_min, y_max, z_min, z_max, max_size);
    let mut nnps = NNPS::new(no_entites, &world_bound, dim);

    // create a 2d domain but with random numbers
    let (mut x, mut y, mut z) = (vec![], vec![], vec![]);
    for _ in 0..no_particles {
        x.push(rng.gen_range(x_min, x_max));
        y.push(rng.gen_range(y_min, y_max));
        z.push(rng.gen_range(z_min, z_max));
    }

    println!("nnps cells number {:?}", nnps.cells.len());

    x[0] = -100.;
    x[1] = 100.;
    y[2] = 100.;
    y[3] = 100.;
    x[4] = -100.;
    y[4] = -100.;
    let mut smpl = Simple::new_fromxy(x.clone(), y.clone(), 0);
    smpl.z = z.clone();

    // stash the particles
    stash_3d(vec![&smpl], &mut nnps);

    for i in (0..x.len()).step_by(5) {
        // get the neighbours from nnps
        let nbrs = get_neighbours_3d(x[i], y[i], z[i], smpl.nnps_idx, &nnps);
        println!("Length of pure neighbours {:?}", nbrs.len());
        let mut filtered_nbrs = vec![];
        // select the neighbours which are in limit or radius scale.
        for &j in nbrs.iter() {
            let rij =
                ((x[i] - x[j]).powf(2.) + (y[i] - y[j]).powf(2.) + (z[i] - z[j]).powf(2.)).sqrt();
            if rij < radius {
                filtered_nbrs.push(j);
            }
        }
        // sort the filtered neighbours
        filtered_nbrs.sort();
        println!("Length of filtered neighbours {:?}", filtered_nbrs.len());

        // compute neighbours by brute force
        let mut brute_nbrs = vec![];
        for j in 0..x.len() {
            let rij =
                ((x[i] - x[j]).powf(2.) + (y[i] - y[j]).powf(2.) + (z[i] - z[j]).powf(2.)).sqrt();
            if rij < radius {
                if (x[i] >= nnps.x_min && x[i] <= nnps.x_max)
                    && (y[i] >= nnps.y_min && y[i] <= nnps.y_max)
                    && (z[i] >= nnps.z_min && z[i] <= nnps.z_max)
                {
                    brute_nbrs.push(j);
                }
            }
        }
        brute_nbrs.sort();
        println!("-------------------------------",);

        assert_eq!(filtered_nbrs, brute_nbrs);
    }
}
// println!("-------------------------------",);
// println!("Length of brute neighbours {:?}", brute_nbrs.len());
// println!("Length of filtered neighbours {:?}", filtered_nbrs.len());
// println!("Length of pure neighbours {:?}", nbrs.len());

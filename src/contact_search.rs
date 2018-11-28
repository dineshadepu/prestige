use physics::dem::DEM;

#[cfg(test)]
use simple_shapes::arange;

/// A block in the simulation world
///
/// Saves the indices of the partices which are in
/// inside it.
///
/// # Example
///
/// Basic usage:
///
/// ```
/// # use prestige::contact_search::Cell;
/// // Define the number of entites world comprises of
/// let no_entities = 3;
///
/// // Create a cell
/// let mut cell = Cell::new(no_entities);
///
/// // Each entity in the world has a nnps_id, so a particle
/// // which belongs to nnps_id of 2 will be saved in `cell.indices[2]`
/// let nnps_id = 2;
/// let idx = 32;
/// cell.indices[nnps_id].push(idx);
///
/// assert_eq!(cell.indices.len(), 3);
/// assert_eq!(cell.indices[2], vec![32]);
/// ```
#[derive(Clone, Debug)]
pub struct Cell {
    /// indices which will be saved in
    pub indices: Vec<Vec<usize>>,
}

impl Cell {
    /// Creates a new cell from given entites
    pub fn new(total_entities: usize) -> Self {
        Cell {
            indices: vec![vec![]; total_entities],
        }
    }
}

/// Dimensions of the world
///
/// It contains information of the minimum, maximum,
/// size of each cell, number of cells and cells of the world
///
///
/// Basic usage:
/// A function `stash` is used to create an NNPS Object
pub struct NNPS {
    pub x_min: f32,
    pub x_max: f32,
    pub y_min: f32,
    pub y_max: f32,
    pub no_x_cells: usize,
    pub no_y_cells: usize,
    pub max_size: f32,
    pub cells: Vec<Cell>,
}

/**
A trait to make the `stash` function generic. This trait is used to
get the reference of to the fields of the struct `x, y, h`. Since
to evaluate neighbours or to store the particle indices in a cell
all we need is position and the size of the particle.

A simple macro is created to implement this trait. So any entity or `struct`
which has fields `x, y, h` can implement this trait by simply executing

```

impl_GetXYH!(DEM)

```

One can check `dem` module for this macro usage.

 **/
pub trait GetXYH {
    /// Get the `x` field on the struct
    fn get_x(&self) -> &[f32];
    /// Get the `y` field on the struct
    fn get_y(&self) -> &[f32];
    /// Get the `h` field on the struct
    fn get_h(&self) -> &[f32];
    /// Get the `nnps_id` field on the struct.
    fn get_nnps_id(&self) -> usize;
    /// Get the `(x, y, h)` fields on the struct.
    fn get_xyh(&self) -> (&[f32], &[f32], &[f32]);
}

#[macro_export]
macro_rules! impl_GetXYH{
    ($($t:ty)*) => ($(
        impl GetXYH for $t {
            fn get_x(&self) ->  &[f32]{
                &self.x
            }
            fn get_y(&self) ->  &[f32]{
                &self.x
            }
            fn get_h(&self) ->  &[f32]{
                &self.x
            }
            fn get_nnps_id(&self) ->  usize{
                self.nnps_idx
            }
            fn get_xyh(&self) -> (&[f32], &[f32], &[f32]) {
                (&self.x, &self.y, &self.h)
            }
        }
    )*)
}

/**
Saves all the particles in respective blocks `Cell` of world.

Given all entites (`DEM` `SPH` `ParticleArray`) in the simulation, finds the
minimum maximum point and also the maximum size of all particles. Then
subdivides the domain into many cells with maximum size.

After creation of the cells each particle in the entity is stashed into its
respective cell block. Which is further used by `get_neighbours` function.

 *TODO*: This should be generic. This has to be updated to support kernels.
 **/

pub fn stash<T: GetXYH>(world: Vec<&T>) -> NNPS {
    // maximum size of a particle
    let mut max_size = 0.;
    let mut x_min = std::f32::INFINITY;
    let mut x_max = -std::f32::INFINITY;
    let mut y_min = std::f32::INFINITY;
    let mut y_max = -std::f32::INFINITY;

    for entity in &world {
        let (x, y, h) = entity.get_xyh();
        for i in 0..x.len() {
            if x_min > x[i] {
                x_min = x[i];
            }
            if x_max < x[i] {
                x_max = x[i];
            }
            if y_min > y[i] {
                y_min = y[i];
            }
            if y_max < y[i] {
                y_max = y[i];
            }
            if max_size < h[i] {
                max_size = h[i];
            }
        }
    }
    // now set the size of the cell
    max_size = 2. * max_size;

    // increase the limits of the simulation world
    x_max += max_size;
    x_min -= max_size;
    y_max += max_size;
    y_min -= max_size;

    // number of cells in x direction are
    let no_x_cells = ((x_max - x_min) / max_size) as usize;
    let no_y_cells = ((y_max - y_min) / max_size) as usize;

    // total number of cells are
    let mut cells = vec![Cell::new(world.len()); no_x_cells as usize * no_y_cells as usize];

    // Stash the particles in the requisite cells
    for entity in &world {
        let nnps_id = entity.get_nnps_id();
        let (x, y, _h) = entity.get_xyh();
        for i in 0..x.len() {
            let x_index = ((x[i] - x_min) / max_size) as usize;
            let y_index = ((y[i] - y_min) / max_size) as usize;

            // one dimentional index is
            let cell_no = x_index + no_x_cells * y_index;
            cells[cell_no].indices[nnps_id].push(i);
        }
    }

    NNPS {
        x_min,
        x_max,
        y_min,
        y_max,
        no_x_cells,
        no_y_cells,
        max_size,
        cells,
    }
}

pub fn get_neighbours(xi: f32, yi: f32, nnps_idx: usize, nnps: &NNPS) -> Vec<usize> {
    // get the cell index of the particle in the simulation world
    let x_index = ((xi - nnps.x_min) / nnps.max_size) as usize;
    let y_index = ((yi - nnps.y_min) / nnps.max_size) as usize;

    // one dimentional index is
    let cell_no = x_index + nnps.no_x_cells * y_index;

    let mut nbrs = vec![];

    // loop over neighbouring cells and copy the neighbours to a vector
    let cells = &nnps.cells;
    for neighbour in &[
        Some(cell_no),
        cell_no.checked_sub(1),
        cell_no.checked_add(1),
        cell_no.checked_sub(nnps.no_x_cells),
        cell_no.checked_sub(nnps.no_x_cells - 1),
        cell_no.checked_sub(nnps.no_x_cells + 1),
        cell_no.checked_add(nnps.no_x_cells),
        cell_no.checked_add(nnps.no_x_cells - 1),
        cell_no.checked_add(nnps.no_x_cells + 1),
    ] {
        if let Some(cell) = neighbour.and_then(|index| cells.get(index)) {
            for &idx in &cell.indices[nnps_idx] {
                nbrs.push(idx)
            }
        }
    }

    nbrs
}

#[cfg(test)]
mod tests {
    use super::*;
    use fluid::prelude::*;

    #[test]
    fn test_nnps_with_single_entity() {
        // create a world with two entities
        let spacing = 0.5;
        let x = arange(0., 5.2, spacing);
        let num = x.len();
        let mut ent1 = DEM {
            x: x,
            y: vec![0.; num],
            r: vec![spacing / 2.; num],
            fx: vec![0.; num],
            fy: vec![0.; num],
            h: vec![spacing / 2.; num],
            nnps_idx: 0,
        };
        let nnps = stash(vec![&ent1]);

        // check the dimensions of the simulation
        (nnps.x_max).should().be_equal_to(5. + 2. * ent1.h[0]);
        (nnps.x_min).should().be_equal_to(0. - 2. * ent1.h[0]);
        (nnps.y_max).should().be_equal_to(2. * ent1.h[0]);
        (nnps.y_min).should().be_equal_to(-2. * ent1.h[0]);
        (nnps.no_x_cells)
            .should()
            .be_equal_to(((5.0 + 2. * ent1.h[0] - (-2. * ent1.h[0])) / (2. * ent1.h[0])) as usize);
        (nnps.no_y_cells)
            .should()
            .be_equal_to(((2. * ent1.h[0] - (-2. * ent1.h[0])) / (2. * ent1.h[0])) as usize);
        (nnps.cells.len()).should().be_equal_to(12 * 2);

        // check the particles stashed in cells
        {
            let cells = &nnps.cells;
            cells[0].indices[0].len().should().be_equal_to(0);
            cells[1].indices[0].len().should().be_equal_to(0);
            cells[2].indices[0].len().should().be_equal_to(0);
            cells[3].indices[0].len().should().be_equal_to(0);
            cells[4].indices[0].len().should().be_equal_to(0);
            cells[5].indices[0].len().should().be_equal_to(0);
            cells[6].indices[0].len().should().be_equal_to(0);
            cells[7].indices[0].len().should().be_equal_to(0);
            cells[8].indices[0].len().should().be_equal_to(0);
            cells[9].indices[0].len().should().be_equal_to(0);
            cells[10].indices[0].len().should().be_equal_to(0);
            cells[11].indices[0].len().should().be_equal_to(0);
            cells[12].indices[0].len().should().be_equal_to(0);
            cells[13].indices[0].len().should().be_equal_to(1);
            cells[14].indices[0].len().should().be_equal_to(1);
            cells[15].indices[0].len().should().be_equal_to(1);
            cells[16].indices[0].len().should().be_equal_to(1);
            cells[17].indices[0].len().should().be_equal_to(1);
            cells[18].indices[0].len().should().be_equal_to(1);
            cells[19].indices[0].len().should().be_equal_to(1);
            cells[20].indices[0].len().should().be_equal_to(1);
            cells[21].indices[0].len().should().be_equal_to(1);
            cells[22].indices[0].len().should().be_equal_to(1);
            cells[23].indices[0].len().should().be_equal_to(1);
        }

        // Check particle neighbours
        let nbrs_1 = get_neighbours(ent1.x[0], ent1.y[0], 0, &nnps);
    }
}

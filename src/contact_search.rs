/// A trait to be implemented by every nnps model to get the neighbours
pub trait NNPSGeneric {
    fn get_neighbours(&self, x: f32, y: f32, z: f32, nnps_idx: usize) -> Vec<usize>;
    fn get_neighbours_1d(&self, x: f32, nnps_idx: usize) -> Vec<usize>;
    fn get_neighbours_2d(&self, x: f32, y: f32, s_nnps: usize) -> Vec<usize>;
    fn get_neighbours_3d(&self, x: f32, y: f32, z: f32, nnps_idx: usize) -> Vec<usize>;
}

impl NNPSGeneric for NNPS {
    fn get_neighbours(&self, x: f32, y: f32, z: f32, nnps_idx: usize) -> Vec<usize> {
        match self.dim {
            1 => self.get_neighbours_1d(x, nnps_idx),
            2 => self.get_neighbours_2d(x, y, nnps_idx),
            3 => self.get_neighbours_3d(x, y, z, nnps_idx),
            _ => panic!("Check your dimension")
        }
    }
    fn get_neighbours_1d(&self, x: f32, nnps_idx: usize) -> Vec<usize> {
        // get the cell index of the particle in the simulation world
        let x_index = ((x - self.x_min) / self.max_size) as usize;

        // one dimentional index is
        let cell_no = x_index;

        let mut nbrs = vec![];

        if x >= self.x_min && x <= self.x_max {
            // loop over neighbouring cells and copy the neighbours to a vector

            let cells = &self.cells;
            for neighbour in &[
                Some(cell_no),
                cell_no.checked_sub(1),
                cell_no.checked_add(1),
            ] {
                if let Some(cell) = neighbour.and_then(|index| cells.get(index)) {
                    for &idx in &cell.indices[nnps_idx] {
                        nbrs.push(idx)
                    }
                }
            }
        }
        nbrs
    }
    fn get_neighbours_2d(&self, x: f32, y: f32, nnps_idx: usize) -> Vec<usize> {
        let mut nbrs = vec![];

        if (x >= self.x_min && x <= self.x_max) && (y >= self.y_min && y <= self.y_max) {
            // get the cell index of the particle in the simulation world
            let x_index = ((x - self.x_min) / self.max_size) as usize;
            let y_index = ((y - self.y_min) / self.max_size) as usize;

            // one dimentional index is
            let cell_no = x_index + self.no_x_cells * y_index;

            // loop over neighbouring cells and copy the neighbours to a vector

            let cells = &self.cells;
            for neighbour in &[
                Some(cell_no),
                cell_no.checked_sub(1),
                cell_no.checked_add(1),
                cell_no.checked_sub(self.no_x_cells),
                cell_no.checked_sub(self.no_x_cells - 1),
                cell_no.checked_sub(self.no_x_cells + 1),
                cell_no.checked_add(self.no_x_cells),
                cell_no.checked_add(self.no_x_cells - 1),
                cell_no.checked_add(self.no_x_cells + 1),
            ] {
                if let Some(cell) = neighbour.and_then(|index| cells.get(index)) {
                    for &idx in &cell.indices[nnps_idx] {
                        nbrs.push(idx)
                    }
                }
            }
        }
        nbrs
    }
    fn get_neighbours_3d(&self, x: f32, y: f32, z: f32, nnps_idx: usize) -> Vec<usize> {
        let mut nbrs = vec![];

        // loop over neighbouring cells and copy the neighbours to a vector

        if (x >= self.x_min && x <= self.x_max)
            && (y >= self.y_min && y <= self.y_max)
            && (z >= self.z_min && z <= self.z_max)
        {
            // get the cell index of the particle in the simulation world
            let x_index = ((x - self.x_min) / self.max_size) as usize;
            let y_index = ((y - self.y_min) / self.max_size) as usize;
            let z_index = ((z - self.z_min) / self.max_size) as usize;
            let no_x_cells = self.no_x_cells;
            let no_y_cells = self.no_y_cells;

            // one dimentional index is
            let cell_no = x_index as usize
                + no_x_cells * y_index as usize
                + no_x_cells * no_y_cells * z_index as usize;

            let xy_cells = no_x_cells * no_y_cells;

            let cells = &self.cells;
            for neighbour in &[
                Some(cell_no),
                cell_no.checked_sub(1),
                cell_no.checked_add(1),
                cell_no.checked_sub(no_x_cells),
                cell_no.checked_sub(no_x_cells - 1),
                cell_no.checked_sub(no_x_cells + 1),
                cell_no.checked_add(no_x_cells),
                cell_no.checked_add(no_x_cells - 1),
                cell_no.checked_add(no_x_cells + 1),
            ] {
                if let Some(cell) = neighbour.and_then(|index| cells.get(index)) {
                    for &idx in &cell.indices[nnps_idx] {
                        nbrs.push(idx)
                    }
                }
            }

            // for the stack of z = +1
            for neighbour in &[
                cell_no.checked_add(xy_cells),
                cell_no.checked_add(xy_cells - 1),
                cell_no.checked_add(xy_cells + 1),
                cell_no.checked_add(xy_cells - no_y_cells),
                cell_no.checked_add(xy_cells - no_y_cells - 1),
                cell_no.checked_add(xy_cells - no_y_cells + 1),
                cell_no.checked_add(xy_cells + no_y_cells),
                cell_no.checked_add(xy_cells + no_y_cells - 1),
                cell_no.checked_add(xy_cells + no_y_cells + 1),
            ] {
                if let Some(cell) = neighbour.and_then(|index| cells.get(index)) {
                    for &idx in &cell.indices[nnps_idx] {
                        nbrs.push(idx)
                    }
                }
            }

            // for the stack of z = -1
            for neighbour in &[
                cell_no.checked_sub(xy_cells),
                cell_no.checked_sub(xy_cells - 1),
                cell_no.checked_sub(xy_cells + 1),
                cell_no.checked_sub(xy_cells - no_y_cells),
                cell_no.checked_sub(xy_cells - no_y_cells - 1),
                cell_no.checked_sub(xy_cells - no_y_cells + 1),
                cell_no.checked_sub(xy_cells + no_y_cells),
                cell_no.checked_sub(xy_cells + no_y_cells - 1),
                cell_no.checked_sub(xy_cells + no_y_cells + 1),
            ] {
                if let Some(cell) = neighbour.and_then(|index| cells.get(index)) {
                    for &idx in &cell.indices[nnps_idx] {
                        nbrs.push(idx)
                    }
                }
            }
        }

        nbrs
    }
}

/// A block in the simulation world
///
/// Saves the indices of the partices which are in
/// inside it.
///
/// # Example
///
/// Basic usage:
///
/// ```norun
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
#[derive(Debug)]
pub struct NNPS {
    pub x_min: f32,
    pub x_max: f32,
    pub y_min: f32,
    pub y_max: f32,
    pub z_min: f32,
    pub z_max: f32,
    pub no_x_cells: usize,
    pub no_y_cells: usize,
    pub no_z_cells: usize,
    pub max_size: f32,
    pub cells: Vec<Cell>,
    pub world_entities: usize,
    pub dim: usize,
}

pub struct WorldBounds {
    pub x_min: f32,
    pub x_max: f32,
    pub y_min: f32,
    pub y_max: f32,
    pub z_min: f32,
    pub z_max: f32,
    pub max_size: f32,
}

impl WorldBounds {
    pub fn new(
        x_min: f32,
        x_max: f32,
        y_min: f32,
        y_max: f32,
        z_min: f32,
        z_max: f32,
        max_size: f32,
    ) -> WorldBounds {
        WorldBounds {
            x_min,
            x_max,
            y_min,
            y_max,
            z_min,
            z_max,
            max_size,
        }
    }
}

impl NNPS {
    pub fn new(world_entities_len: usize, bounds: &WorldBounds, dim: usize) -> NNPS {
        {
            // maximum size of a particle
            let x_min = bounds.x_min;
            let x_max = bounds.x_max;
            let y_min = bounds.y_min;
            let y_max = bounds.y_max;
            let z_min = bounds.z_min;
            let z_max = bounds.z_max;
            let max_size = bounds.max_size;

            let no_x_cells = ((x_max + max_size - x_min) / max_size) as usize;
            let (no_y_cells, no_z_cells) = match dim {
                1 => (1, 1),
                2 => {
                    if y_max < y_min {
                        panic!("Check world bounds. Y_MAX is less than Y_MIN")
                    } else {
                        (((y_max + max_size - y_min) / max_size) as usize, 1)
                    }
                }
                3 => {
                    if y_max < y_min {
                        panic!("Check world bounds. Y_MAX is less than Y_MIN")
                    } else {
                        (
                            ((y_max + max_size - y_min) / max_size) as usize,
                            ((z_max + max_size - z_min) / max_size) as usize,
                        )
                    }
                }
                _ => panic!("We don't support the current dimension"),
            };

            // total number of cells are
            let cells = vec![Cell::new(world_entities_len); no_x_cells * no_y_cells * no_z_cells];

            NNPS {
                x_min,
                x_max,
                y_min,
                y_max,
                z_min,
                z_max,
                no_x_cells,
                no_y_cells,
                no_z_cells,
                max_size,
                cells,
                world_entities: world_entities_len,
                dim,
            }
        }
    }
}

/**
A trait to make the `stash` function generic. This trait is used to
get the reference of to the fields of the struct `x, y, h`. Since
to evaluate neighbours or to store the particle indices in a cell
all we need is position and the size of the particle.

A simple macro is created to implement this trait. So any entity or `struct`
which has fields `x, y, h` can implement this trait by simply executing

```norun

impl_GetXYZH!(DEM)

```

One can check `dem` module for this macro usage.

 **/
pub trait GetXYZH {
    /// Get the `nnps_id` field on the struct.
    fn get_nnps_id(&self) -> usize;
    /// Get the `(x, y, h)` fields on the struct.
    fn get_xyzh(&self) -> (&[f32], &[f32], &[f32], &[f32]);
}

#[macro_export]
macro_rules! impl_GetXYZH{
    ($($t:ty)*) => ($(
        impl GetXYZH for $t {
            fn get_nnps_id(&self) ->  usize{
                self.nnps_idx
            }
            fn get_xyzh(&self) -> (&[f32], &[f32], &[f32], &[f32]) {
                (&self.x, &self.y, &self.z, &self.h)
            }
        }
    )*)
}

pub fn stash_1d<T: GetXYZH>(world: Vec<&T>, nnps: &mut NNPS) {
    let x_max = nnps.x_max;
    let x_min = nnps.x_min;
    let max_size = nnps.max_size;
    let world_entities = nnps.world_entities;

    let cells = &mut nnps.cells;

    // clean the cells
    for i in 0..cells.len() {
        for j in 0..world_entities {
            cells[i].indices[j].clear()
        }
    }
    // Stash the particles in the requisite cells
    for entity in &world {
        let nnps_id = entity.get_nnps_id();
        let (x, _y, _z, _h) = entity.get_xyzh();
        for i in 0..x.len() {
            // check if the particle is in nnps domain
            if x[i] >= x_min && x[i] <= x_max {
                let x_index = (x[i] - x_min) / max_size;

                // one dimentional index is
                let cell_no = x_index as usize;
                cells[cell_no].indices[nnps_id].push(i);
            }
        }
    }
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

pub fn stash_2d<T: GetXYZH>(world: Vec<&T>, nnps: &mut NNPS) {
    let x_max = nnps.x_max;
    let x_min = nnps.x_min;
    let y_max = nnps.y_max;
    let y_min = nnps.y_min;
    let no_x_cells = nnps.no_x_cells;
    let max_size = nnps.max_size;
    let world_entities = nnps.world_entities;

    let cells = &mut nnps.cells;

    // clean the cells
    for i in 0..cells.len() {
        for j in 0..world_entities {
            cells[i].indices[j].clear()
        }
    }
    // Stash the particles in the requisite cells
    for entity in &world {
        let nnps_id = entity.get_nnps_id();
        let (x, y, _z, _h) = entity.get_xyzh();
        for i in 0..x.len() {
            // check if the particle is in nnps domain
            if (x[i] >= x_min && x[i] <= x_max) && (y[i] >= y_min && y[i] <= y_max) {
                let x_index = (x[i] - x_min) / max_size;
                let y_index = (y[i] - y_min) / max_size;

                // one dimentional index is
                let cell_no = x_index as usize + no_x_cells * y_index as usize;
                cells[cell_no].indices[nnps_id].push(i);
            }
        }
    }
}

pub fn stash_3d<T: GetXYZH>(world: Vec<&T>, nnps: &mut NNPS) {
    let x_min = nnps.x_min;
    let y_min = nnps.y_min;
    let z_min = nnps.z_min;
    let x_max = nnps.x_max;
    let y_max = nnps.y_max;
    let z_max = nnps.z_max;
    let no_x_cells = nnps.no_x_cells;
    let no_y_cells = nnps.no_y_cells;
    let max_size = nnps.max_size;
    let world_entities = nnps.world_entities;

    let cells = &mut nnps.cells;

    // clean the cells
    for i in 0..cells.len() {
        for j in 0..world_entities {
            cells[i].indices[j].clear()
        }
    }
    // Stash the particles in the requisite cells
    for entity in &world {
        let nnps_id = entity.get_nnps_id();
        let (x, y, z, _h) = entity.get_xyzh();
        for i in 0..x.len() {
            // check if the particle is in nnps domain
            if (x[i] >= x_min && x[i] <= x_max)
                && (y[i] >= y_min && y[i] <= y_max)
                && (z[i] >= z_min && z[i] <= z_max)
            {
                let x_index = (x[i] - x_min) / max_size;
                let y_index = (y[i] - y_min) / max_size;
                let z_index = (z[i] - z_min) / max_size;

                let cell_no = x_index as usize
                    + no_x_cells * y_index as usize
                    + no_x_cells * no_y_cells * z_index as usize;

                cells[cell_no].indices[nnps_id].push(i);
            }
        }
    }
}

pub fn get_neighbours_1d(xi: f32, _yi: f32, _zi: f32, nnps_idx: usize, nnps: &NNPS) -> Vec<usize> {
    // get the cell index of the particle in the simulation world
    let x_index = ((xi - nnps.x_min) / nnps.max_size) as usize;

    // one dimentional index is
    let cell_no = x_index;

    let mut nbrs = vec![];

    if xi >= nnps.x_min && xi <= nnps.x_max {
        // loop over neighbouring cells and copy the neighbours to a vector

        let cells = &nnps.cells;
        for neighbour in &[
            Some(cell_no),
            cell_no.checked_sub(1),
            cell_no.checked_add(1),
        ] {
            if let Some(cell) = neighbour.and_then(|index| cells.get(index)) {
                for &idx in &cell.indices[nnps_idx] {
                    nbrs.push(idx)
                }
            }
        }
    }
    nbrs
}

pub fn get_neighbours_2d(xi: f32, yi: f32, _zi: f32, nnps_idx: usize, nnps: &NNPS) -> Vec<usize> {
    let mut nbrs = vec![];

    if (xi >= nnps.x_min && xi <= nnps.x_max) && (yi >= nnps.y_min && yi <= nnps.y_max) {
        // get the cell index of the particle in the simulation world
        let x_index = ((xi - nnps.x_min) / nnps.max_size) as usize;
        let y_index = ((yi - nnps.y_min) / nnps.max_size) as usize;

        // one dimentional index is
        let cell_no = x_index + nnps.no_x_cells * y_index;

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
    }
    nbrs
}

pub fn get_neighbours_3d(xi: f32, yi: f32, zi: f32, nnps_idx: usize, nnps: &NNPS) -> Vec<usize> {
    let mut nbrs = vec![];

    // loop over neighbouring cells and copy the neighbours to a vector

    if (xi >= nnps.x_min && xi <= nnps.x_max)
        && (yi >= nnps.y_min && yi <= nnps.y_max)
        && (zi >= nnps.z_min && zi <= nnps.z_max)
    {
        // get the cell index of the particle in the simulation world
        let x_index = ((xi - nnps.x_min) / nnps.max_size) as usize;
        let y_index = ((yi - nnps.y_min) / nnps.max_size) as usize;
        let z_index = ((zi - nnps.z_min) / nnps.max_size) as usize;
        let no_x_cells = nnps.no_x_cells;
        let no_y_cells = nnps.no_y_cells;

        // one dimentional index is
        let cell_no = x_index as usize
            + no_x_cells * y_index as usize
            + no_x_cells * no_y_cells * z_index as usize;

        let xy_cells = no_x_cells * no_y_cells;

        let cells = &nnps.cells;
        for neighbour in &[
            Some(cell_no),
            cell_no.checked_sub(1),
            cell_no.checked_add(1),
            cell_no.checked_sub(no_x_cells),
            cell_no.checked_sub(no_x_cells - 1),
            cell_no.checked_sub(no_x_cells + 1),
            cell_no.checked_add(no_x_cells),
            cell_no.checked_add(no_x_cells - 1),
            cell_no.checked_add(no_x_cells + 1),
        ] {
            if let Some(cell) = neighbour.and_then(|index| cells.get(index)) {
                for &idx in &cell.indices[nnps_idx] {
                    nbrs.push(idx)
                }
            }
        }

        // for the stack of z = +1
        for neighbour in &[
            cell_no.checked_add(xy_cells),
            cell_no.checked_add(xy_cells - 1),
            cell_no.checked_add(xy_cells + 1),
            cell_no.checked_add(xy_cells - no_y_cells),
            cell_no.checked_add(xy_cells - no_y_cells - 1),
            cell_no.checked_add(xy_cells - no_y_cells + 1),
            cell_no.checked_add(xy_cells + no_y_cells),
            cell_no.checked_add(xy_cells + no_y_cells - 1),
            cell_no.checked_add(xy_cells + no_y_cells + 1),
        ] {
            if let Some(cell) = neighbour.and_then(|index| cells.get(index)) {
                for &idx in &cell.indices[nnps_idx] {
                    nbrs.push(idx)
                }
            }
        }

        // for the stack of z = -1
        for neighbour in &[
            cell_no.checked_sub(xy_cells),
            cell_no.checked_sub(xy_cells - 1),
            cell_no.checked_sub(xy_cells + 1),
            cell_no.checked_sub(xy_cells - no_y_cells),
            cell_no.checked_sub(xy_cells - no_y_cells - 1),
            cell_no.checked_sub(xy_cells - no_y_cells + 1),
            cell_no.checked_sub(xy_cells + no_y_cells),
            cell_no.checked_sub(xy_cells + no_y_cells - 1),
            cell_no.checked_sub(xy_cells + no_y_cells + 1),
        ] {
            if let Some(cell) = neighbour.and_then(|index| cells.get(index)) {
                for &idx in &cell.indices[nnps_idx] {
                    nbrs.push(idx)
                }
            }
        }
    }

    nbrs
}

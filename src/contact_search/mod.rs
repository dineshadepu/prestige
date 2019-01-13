pub mod hash_nnps;
pub mod linked_nnps;

#[cfg(test)]
pub mod test_hash_nnps;
pub mod test_collision_detection;

/// A trait to be implemented by every nnps model to get the neighbours
pub trait NNPSGeneric {
    fn get_neighbours(&self, x: f32, y: f32, z: f32, nnps_idx: usize) -> Vec<usize>;
    fn get_neighbours_1d(&self, x: f32, nnps_idx: usize) -> Vec<usize>;
    fn get_neighbours_2d(&self, x: f32, y: f32, s_nnps: usize) -> Vec<usize>;
    fn get_neighbours_3d(&self, x: f32, y: f32, z: f32, nnps_idx: usize) -> Vec<usize>;
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

pub fn brute_force_nbrs<T: GetXYZH>(
    (xi, yi, zi): (f32, f32, f32),
    radius: f32,
    entity: &T,
) -> Vec<usize> {
    let (x, y, z, _) = entity.get_xyzh();
    let mut nbrs = vec![];
    for i in 0..x.len() {
        let dist = (xi - x[i]).powf(2.) + (yi - y[i]).powf(2.) + (zi - z[i]).powf(2.);
        if dist < radius {
            nbrs.push(i);
        }
    }
    nbrs
}

pub mod benchmarks;
pub use itertools_num::linspace;


use std::path::PathBuf;
use vtkio::export_ascii;
use vtkio::model::*;

pub fn grid_linspace(
    xl: f32,
    xr: f32,
    xnum: usize,
    yl: f32,
    yr: f32,
    ynum: usize,
) -> (Vec<f32>, Vec<f32>) {
    // create x range
    let x = linspace::<f32>(xl, xr, xnum).collect::<Vec<_>>();

    // create y range
    let y = linspace::<f32>(yl, yr, ynum).collect::<Vec<_>>();

    let mut x_grid = vec![];
    let mut y_grid = vec![];

    for i in 0..y.len() {
        for j in 0..x.len() {
            x_grid.push(x[j]);
            y_grid.push(y[i]);
        }
    }
    (x_grid, y_grid)
}

pub fn arange(left: f32, right: f32, step: f32) -> Vec<f32> {
    let mut x = vec![];
    let mut tmp = left;

    while tmp < right {
        x.push(tmp);
        tmp += step;
    }
    x
}

#[test]
fn test_arange() {
    assert_eq!(vec![0., 1., 2., 3., 4.], arange(0., 5., 1.));
}

pub fn grid_arange(
    xl: f32,
    xr: f32,
    x_spacing: f32,
    yl: f32,
    yr: f32,
    y_spacing: f32,
) -> (Vec<f32>, Vec<f32>) {
    let x_arange = arange(xl, xr, x_spacing);
    let y_arange = arange(yl, yr, y_spacing);

    let mut x_grid = vec![];
    let mut y_grid = vec![];

    for i in 0..y_arange.len() {
        for j in 0..x_arange.len() {
            x_grid.push(x_arange[j]);
            y_grid.push(y_arange[i]);
        }
    }
    (x_grid, y_grid)
}

pub fn grid_arange_3d(
    xl: f32,
    xr: f32,
    x_spacing: f32,
    yl: f32,
    yr: f32,
    y_spacing: f32,
    zl: f32,
    zr: f32,
    z_spacing: f32,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let x_arange = arange(xl, xr, x_spacing);
    let y_arange = arange(yl, yr, y_spacing);
    let z_arange = arange(zl, zr, z_spacing);

    let mut x_grid = vec![];
    let mut y_grid = vec![];
    let mut z_grid = vec![];

    for k in 0..z_arange.len() {
        for i in 0..y_arange.len() {
            for j in 0..x_arange.len() {
                x_grid.push(x_arange[j]);
                y_grid.push(y_arange[i]);
                z_grid.push(z_arange[k]);
            }
        }
    }
    (x_grid, y_grid, z_grid)
}

pub fn tank_2d(
    mut xl: f32,
    mut xr: f32,
    x_spacing: f32,
    mut yl: f32,
    mut yr: f32,
    y_spacing: f32,
    layers: usize,
    outside: bool,
) -> (Vec<f32>, Vec<f32>) {
    assert!(layers > 0);
    if outside {
        xl = xl - (layers - 1) as f32 * x_spacing - x_spacing / 2.;
        xr = xr + (layers - 1) as f32 * x_spacing + x_spacing / 2.;
        yl = yl - (layers - 1) as f32 * y_spacing - y_spacing / 2.;
        yr = yr + (layers - 1) as f32 * y_spacing + y_spacing / 2.;
    }
    let x_arange = arange(xl, xr, x_spacing);

    let (xg, yg) = grid_arange(xl, xr, x_spacing, yl, yr, y_spacing);

    // now filter the particles which only belong to tank
    let (mut x, mut y) = (vec![], vec![]);

    let x_left_cutoff = xl + (layers - 1) as f32 * x_spacing + x_spacing / 2.;
    let x_right_cutoff =
        x_arange[x_arange.len() - 1] - (layers - 1) as f32 * x_spacing - x_spacing / 2.;
    let y_bottom_cutoff = yl + (layers - 1) as f32 * y_spacing + y_spacing / 2.;

    for i in 0..xg.len() {
        if xg[i] < x_left_cutoff || xg[i] > x_right_cutoff {
            x.push(xg[i]);
            y.push(yg[i]);
        } else if yg[i] < y_bottom_cutoff {
            x.push(xg[i]);
            y.push(yg[i]);
        }
    }
    (x, y)
}

pub fn tank_3d(
    xl: f32,
    xr: f32,
    x_spacing: f32,
    yl: f32,
    yr: f32,
    y_spacing: f32,
    zl: f32,
    zr: f32,
    z_spacing: f32,
    layers: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let x_arange = arange(xl, xr, x_spacing);
    let z_arange = arange(zl, zr, z_spacing);

    let (xg, yg, zg) = grid_arange_3d(xl, xr, x_spacing, yl, yr, y_spacing, zl, zr, z_spacing);

    // now filter the particles which only belong to tank
    let (mut x, mut y, mut z) = (vec![], vec![], vec![]);

    let x_left_cutoff = xl + (layers - 1) as f32 * x_spacing + x_spacing / 2.;
    let x_right_cutoff =
        x_arange[x_arange.len() - 1] - (layers - 1) as f32 * x_spacing - x_spacing / 2.;
    let y_bottom_cutoff = yl + (layers - 1) as f32 * y_spacing + y_spacing / 2.;
    let z_back_cutoff = zl + (layers - 1) as f32 * z_spacing + z_spacing / 2.;
    let z_front_cutoff =
        z_arange[z_arange.len() - 1] - (layers - 1) as f32 * z_spacing - z_spacing / 2.;

    for i in 0..xg.len() {
        if xg[i] < x_left_cutoff || xg[i] > x_right_cutoff {
            x.push(xg[i]);
            y.push(yg[i]);
            z.push(zg[i]);
        } else if yg[i] < y_bottom_cutoff || zg[i] < z_back_cutoff || zg[i] > z_front_cutoff {
            x.push(xg[i]);
            y.push(yg[i]);
            z.push(zg[i]);
        }
    }
    (x, y, z)
}

pub fn hollow_box_2d(
    mut xl: f32,
    mut xr: f32,
    x_spacing: f32,
    mut yl: f32,
    mut yr: f32,
    y_spacing: f32,
    layers: usize,
    outside: bool,
) -> (Vec<f32>, Vec<f32>) {
    let (xl_lim, xr_lim, yl_lim, yr_lim) = match outside {
        true => {
            let xl_lim = xl;
            let xr_lim = xr;
            let yl_lim = yl;
            let yr_lim = yr;
            xl = xl - layers as f32 * x_spacing;
            yl = yl - layers as f32 * y_spacing;
            xr = xr + layers as f32 * x_spacing + x_spacing / 2.;
            yr = yr + layers as f32 * y_spacing + y_spacing / 2.;
            (xl_lim, xr_lim, yl_lim, yr_lim)
        }
        false => {
            let xl_lim = xl + (layers - 1) as f32 * x_spacing + x_spacing / 2.;
            let xr_lim = xr - (layers - 1) as f32 * x_spacing - x_spacing / 2.;
            let yl_lim = yl + (layers - 1) as f32 * y_spacing + y_spacing / 2.;
            let yr_lim = yr - (layers - 1) as f32 * y_spacing - y_spacing / 2.;
            (xl_lim, xr_lim, yl_lim, yr_lim)
        }
    };
    // let x_arange = arange(xl, xr, x_spacing);

    let (xg, yg) = grid_arange(xl, xr, x_spacing, yl, yr, y_spacing);

    let (mut x, mut y) = (vec![], vec![]);

    for i in 0..xg.len() {
        if xg[i] < xl_lim || xg[i] > xr_lim {
            x.push(xg[i]);
            y.push(yg[i]);
        } else if yg[i] < yl_lim || yg[i] > yr_lim {
            x.push(xg[i]);
            y.push(yg[i]);
        }
    }
    (x, y)
}

pub fn circle_2d(center: (f32, f32), radius: f32, spacing: f32) -> (Vec<f32>, Vec<f32>) {
    // create a 2d grid
    let (xg, yg) = grid_arange(
        center.0 - radius + spacing / 2.,
        center.0 + radius + spacing / 2.,
        spacing,
        center.1 - radius + spacing / 2.,
        center.1 + radius + spacing / 2.,
        spacing,
    );

    // filter the particles which are out of the circle
    let mut xc = vec![];
    let mut yc = vec![];
    for i in 0..xg.len() {
        if (xg[i] - center.0).powf(2.0) + (yg[i] - center.1).powf(2.0) <= radius.powf(2.) {
            xc.push(xg[i]);
            yc.push(yg[i]);
        }
    }
    (xc, yc)
}

/// A simple struct to test different geometries
pub struct Entity {
    x: Vec<f32>,
    y: Vec<f32>,
    z: Vec<f32>,
    rad: Vec<f32>,
}

impl Entity {
    pub fn from_xyz_rad(x: Vec<f32>, y: Vec<f32>, z: Vec<f32>, rad: Vec<f32>) -> Self {
        Entity { x, y, z, rad }
    }

    pub fn from_xy(x: Vec<f32>, y: Vec<f32>) -> Self {
        Entity::from_xyz_rad(x.clone(), y.clone(), vec![0.; x.len()], vec![1.; x.len()])
    }
    pub fn write_vtk(self, filename: &str) {
        let mut pos = vec![];
        let mut radius = vec![];
        for i in 0..self.x.len() {
            pos.push(self.x[i]);
            pos.push(self.y[i]);
            pos.push(self.z[i]);
            radius.push(self.rad[i]);
        }

        let mut attributes = Attributes::new();
        attributes.point.push((
            "Radius".to_string(),
            Attribute::Scalars {
                num_comp: 1,
                lookup_table: None,
                data: radius.into(),
            },
        ));

        let data = DataSet::UnstructuredGrid {
            points: pos.into(),
            cells: Cells {
                num_cells: 0,
                vertices: vec![],
            },
            cell_types: vec![],
            data: attributes,
        };

        let vtk = Vtk {
            version: Version::new((4, 1)),
            title: String::from("Data"),
            data: data,
        };

        let _p = export_ascii(vtk, &PathBuf::from(&filename));
    }
}

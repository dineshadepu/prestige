// sub modules
pub mod dem_linear;
pub mod dem_nonlinear;
pub mod equations;

// local library imports
use crate::WriteOutput;
// import to implement this trait for nnps functionality
use crate::contact_search::GetXYZH;

// std library imports
use std::path::PathBuf;

// io library
use vtkio::export_ascii;
use vtkio::model::*;

pub struct DEM {
    pub x: Vec<f32>,
    pub y: Vec<f32>,
    pub z: Vec<f32>,
    pub x0: Vec<f32>,
    pub y0: Vec<f32>,
    pub z0: Vec<f32>,
    pub u: Vec<f32>,
    pub v: Vec<f32>,
    pub w: Vec<f32>,
    pub u0: Vec<f32>,
    pub v0: Vec<f32>,
    pub w0: Vec<f32>,
    pub r: Vec<f32>,
    pub m: Vec<f32>,
    pub fx: Vec<f32>,
    pub fy: Vec<f32>,
    pub fz: Vec<f32>,
    pub h: Vec<f32>,
    pub nnps_idx: usize,
    pub no_par: usize,
}

impl DEM {
    pub fn new_from_xyzh(x: Vec<f32>, y: Vec<f32>, h: Vec<f32>, no_par: usize) -> Self {
        DEM {
            x: x,
            y: y,
            h: h,
            z: vec![0.; no_par],
            x0: vec![0.; no_par],
            y0: vec![0.; no_par],
            z0: vec![0.; no_par],
            u: vec![0.; no_par],
            v: vec![0.; no_par],
            w: vec![0.; no_par],
            u0: vec![0.; no_par],
            v0: vec![0.; no_par],
            w0: vec![0.; no_par],
            r: vec![0.; no_par],
            fx: vec![0.; no_par],
            fy: vec![0.; no_par],
            fz: vec![0.; no_par],
            nnps_idx: 0,
            no_par: no_par,
            m: vec![0.; no_par],
        }
    }
}

impl WriteOutput for DEM {
    fn write_vtk(&self, output: String) {
        let x = &self.x;
        let y = &self.y;
        let z = &self.z;
        let u = &self.u;
        let v = &self.v;
        let w = &self.w;
        let r = &self.r;
        let fx = &self.fx;
        let fy = &self.fy;
        let fz = &self.fz;

        let mut pos = vec![];
        let mut vel = vec![];
        let mut force = vec![];
        let mut radius = vec![];
        for i in 0..x.len() {
            pos.push(x[i]);
            pos.push(y[i]);
            pos.push(z[i]);
            vel.push(u[i]);
            vel.push(v[i]);
            vel.push(w[i]);
            force.push(fx[i]);
            force.push(fy[i]);
            force.push(fz[i]);
            radius.push(r[i]);
        }

        let mut attributes = Attributes::new();
        attributes.point.push((
            "Force".to_string(),
            Attribute::Vectors { data: force.into() },
        ));
        attributes.point.push((
            "Velocity".to_string(),
            Attribute::Vectors { data: vel.into() },
        ));
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

        let _p = export_ascii(vtk, &PathBuf::from(&output));
    }
}

// implement nnps macro
impl_GetXYZH!(DEM);

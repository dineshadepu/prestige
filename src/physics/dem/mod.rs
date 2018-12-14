// sub modules
pub mod dem_linear;
pub mod dem_nonlinear;
pub mod equations;

// local library imports
use crate::WriteOutput;
// import to implement this trait for nnps functionality
use crate::contact_search::GetXYH;

// std library imports
use std::path::PathBuf;

// io library
use vtkio::model::*;
use vtkio::{export_ascii};

pub struct DEM {
    pub x: Vec<f32>,
    pub y: Vec<f32>,
    pub x0: Vec<f32>,
    pub y0: Vec<f32>,
    pub u: Vec<f32>,
    pub v: Vec<f32>,
    pub u0: Vec<f32>,
    pub v0: Vec<f32>,
    pub r: Vec<f32>,
    pub m: Vec<f32>,
    pub fx: Vec<f32>,
    pub fy: Vec<f32>,
    pub h: Vec<f32>,
    pub nnps_idx: usize,
    pub no_par: usize,
}

impl WriteOutput for DEM {
    fn write_vtk(&self, output: String) {
        let x = &self.x;
        let y = &self.y;
        let u = &self.u;
        let v = &self.v;
        let r = &self.r;
        let fx = &self.fx;
        let fy = &self.fy;

        let mut pos = vec![];
        let mut vel = vec![];
        let mut force = vec![];
        let mut radius = vec![];
        for i in 0..x.len() {
            pos.push(x[i]);
            pos.push(y[i]);
            pos.push(0.);
            vel.push(u[i]);
            vel.push(v[i]);
            vel.push(0.);
            force.push(fx[i]);
            force.push(fy[i]);
            force.push(0.);
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
impl_GetXYH!(DEM);

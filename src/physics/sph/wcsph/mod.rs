// sub modules
#[macro_use] pub mod equations;
pub mod test_equations;

// local library imports
use crate::WriteOutput;
// import to implement this trait for nnps functionality
use crate::contact_search::GetXYH;

// std library imports
use std::path::PathBuf;

// io library
use vtkio::export_ascii;
use vtkio::model::*;

pub struct WCSPH {
    pub x: Vec<f32>,
    pub y: Vec<f32>,
    pub x0: Vec<f32>,
    pub y0: Vec<f32>,
    pub u: Vec<f32>,
    pub v: Vec<f32>,
    pub u0: Vec<f32>,
    pub v0: Vec<f32>,
    pub m: Vec<f32>,
    pub rho: Vec<f32>,
    pub rho0: Vec<f32>,
    pub arho: Vec<f32>,
    pub au: Vec<f32>,
    pub av: Vec<f32>,
    pub h: Vec<f32>,
    pub nnps_idx: usize,
    pub no_par: usize,
}

impl WCSPH {
    pub fn new_with_xy(x: Vec<f32>, y: Vec<f32>, nnps_idx: usize) -> Self {
        let other_props = vec![0.; x.len()];
        let no_par = x.len();
        WCSPH {
            x: x,
            y: y,
            x0: other_props.clone(),
            y0: other_props.clone(),
            u: other_props.clone(),
            v: other_props.clone(),
            u0: other_props.clone(),
            v0: other_props.clone(),
            m: other_props.clone(),
            rho: other_props.clone(),
            rho0: other_props.clone(),
            arho: other_props.clone(),
            au: other_props.clone(),
            av: other_props.clone(),
            h: other_props.clone(),
            nnps_idx: nnps_idx,
            no_par: no_par,
        }
    }
}

impl WriteOutput for WCSPH {
    fn write_vtk(&self, output: String) {
        let x = &self.x;
        let y = &self.y;
        let u = &self.u;
        let v = &self.v;
        let au = &self.au;
        let av = &self.av;

        let mut pos = vec![];
        let mut vel = vec![];
        let mut accel = vec![];
        for i in 0..x.len() {
            pos.push(x[i]);
            pos.push(y[i]);
            pos.push(0.);
            vel.push(u[i]);
            vel.push(v[i]);
            vel.push(0.);
            accel.push(au[i]);
            accel.push(av[i]);
            accel.push(0.);
        }

        let mut attributes = Attributes::new();
        attributes.point.push((
            "Acceleration".to_string(),
            Attribute::Vectors { data: accel.into() },
        ));
        attributes.point.push((
            "Velocity".to_string(),
            Attribute::Vectors { data: vel.into() },
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
impl_GetXYH!(WCSPH);

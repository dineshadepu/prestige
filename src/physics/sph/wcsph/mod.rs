// sub modules
#[macro_use] pub mod equations;

// local library imports
use crate::WriteOutput;
// import to implement this trait for nnps functionality
use crate::contact_search::GetXYZH;

// std library imports
use std::path::PathBuf;

// io library
use vtkio::export_ascii;
use vtkio::model::*;

pub struct WCSPH {
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
    pub m: Vec<f32>,
    pub rho: Vec<f32>,
    pub rho0: Vec<f32>,
    pub arho: Vec<f32>,
    pub au: Vec<f32>,
    pub av: Vec<f32>,
    pub aw: Vec<f32>,
    pub h: Vec<f32>,
    pub p: Vec<f32>,
    pub c: Vec<f32>,
    pub nnps_idx: usize,
    pub no_par: usize,
}

impl WCSPH {
    pub fn new_with_xyz(x: Vec<f32>, y: Vec<f32>, z: Vec<f32>, nnps_idx: usize) -> Self {
        let other_props = vec![0.; x.len()];
        let no_par = x.len();
        WCSPH {
            x: x,
            y: y,
            z: z,
            x0: other_props.clone(),
            y0: other_props.clone(),
            z0: other_props.clone(),
            u: other_props.clone(),
            v: other_props.clone(),
            w: other_props.clone(),
            u0: other_props.clone(),
            v0: other_props.clone(),
            w0: other_props.clone(),
            m: other_props.clone(),
            rho: other_props.clone(),
            rho0: other_props.clone(),
            arho: other_props.clone(),
            au: other_props.clone(),
            av: other_props.clone(),
            aw: other_props.clone(),
            h: other_props.clone(),
            p: other_props.clone(),
            c: other_props.clone(),
            nnps_idx: nnps_idx,
            no_par: no_par,
        }
    }
}

impl WriteOutput for WCSPH {
    fn write_vtk(&self, output: String) {
        let x = &self.x;
        let y = &self.y;
        let z = &self.z;
        let u = &self.u;
        let v = &self.v;
        let w = &self.w;
        let au = &self.au;
        let av = &self.av;
        let aw = &self.aw;

        let mut pos = vec![];
        let mut vel = vec![];
        let mut accel_vel = vec![];
        let mut h = vec![];
        let mut p = vec![];
        let mut m = vec![];
        let mut rho = vec![];
        let mut arho = vec![];
        for i in 0..x.len() {
            pos.push(x[i]);
            pos.push(y[i]);
            pos.push(z[i]);
            vel.push(u[i]);
            vel.push(v[i]);
            vel.push(w[i]);
            accel_vel.push(au[i]);
            accel_vel.push(av[i]);
            accel_vel.push(aw[i]);
            h.push(self.h[i]);
            p.push(self.p[i]);
            m.push(self.m[i]);
            rho.push(self.rho[i]);
            arho.push(self.arho[i]);
        }

        let mut attributes = Attributes::new();
        attributes.point.push((
            "AccelerationVel".to_string(),
            Attribute::Vectors { data: accel_vel.into() },
        ));
        attributes.point.push((
            "Velocity".to_string(),
            Attribute::Vectors { data: vel.into() },
        ));

        attributes.point.push((
            "h".to_string(),
            Attribute::Scalars {
                num_comp: 1,
                lookup_table: None,
                data: h.into(),
            },
        ));

        attributes.point.push((
            "p".to_string(),
            Attribute::Scalars {
                num_comp: 1,
                lookup_table: None,
                data: p.into(),
            },
        ));
        attributes.point.push((
            "m".to_string(),
            Attribute::Scalars {
                num_comp: 1,
                lookup_table: None,
                data: m.into(),
            },
        ));
        attributes.point.push((
            "rho".to_string(),
            Attribute::Scalars {
                num_comp: 1,
                lookup_table: None,
                data: rho.into(),
            },
        ));

        attributes.point.push((
            "arho".to_string(),
            Attribute::Scalars {
                num_comp: 1,
                lookup_table: None,
                data: arho.into(),
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
impl_GetXYZH!(WCSPH);

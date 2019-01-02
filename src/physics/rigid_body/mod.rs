pub mod equations;

use cgmath::{Matrix3, Vector3};
use cgmath::prelude::*;
use std::collections::HashMap;

// import to implement this trait for nnps functionality
use crate::contact_search::GetXYZH;
use crate::WriteOutput;

// std library imports
use std::path::PathBuf;

// io library
use vtkio::export_ascii;
use vtkio::model::*;

pub struct RB3d {
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
    pub h: Vec<f32>,
    pub m: Vec<f32>,
    pub fx: Vec<f32>,
    pub fy: Vec<f32>,
    pub fz: Vec<f32>,
    pub rad: Vec<f32>,
    pub r_dash: Vec<Vector3<f32>>,
    pub r_dash0: Vec<Vector3<f32>>,
    pub r_body: Vec<Vector3<f32>>,
    pub cm: Vector3<f32>,
    pub lin_vel: Vector3<f32>,
    pub ang_vel: Vector3<f32>,
    pub lin_mom: Vector3<f32>,
    pub ang_mom: Vector3<f32>,
    pub torque: Vector3<f32>,
    pub net_force: Vector3<f32>,
    pub orientation: Matrix3<f32>,
    pub orientation_rate: Matrix3<f32>,
    pub moi_body_inv: Matrix3<f32>,
    pub moi_global_inv: Matrix3<f32>,
    pub total_mass: f32,
    pub nnps_idx: usize,
}

impl RB3d {
    /// Creates a DEMLinear entity from x, y, z and radius of given particles
    pub fn from_xyzr(x: Vec<f32>, y: Vec<f32>, z: Vec<f32>, rad: Vec<f32>) -> RB3d {
        RB3d {
            x: x.clone(),
            y: y,
            z: z,
            rad: rad,
            x0: vec![0.; x.len()],
            y0: vec![0.; x.len()],
            z0: vec![0.; x.len()],
            u: vec![0.; x.len()],
            v: vec![0.; x.len()],
            w: vec![0.; x.len()],
            u0: vec![0.; x.len()],
            v0: vec![0.; x.len()],
            w0: vec![0.; x.len()],
            h: vec![0.; x.len()],
            m: vec![0.; x.len()],
            fx: vec![0.; x.len()],
            fy: vec![0.; x.len()],
            fz: vec![0.; x.len()],
            r_dash: vec![Vector3::zero(); x.len()],
            r_dash0: vec![Vector3::zero(); x.len()],
            r_body: vec![Vector3::zero(); x.len()],
            cm: Vector3::new(0., 0., 0.),
            lin_vel: Vector3::new(0., 0., 0.),
            ang_vel: Vector3::new(0., 0., 0.),
            lin_mom: Vector3::new(0., 0., 0.),
            ang_mom: Vector3::new(0., 0., 0.),
            torque: Vector3::new(0., 0., 0.),
            net_force: Vector3::new(0., 0., 0.),
            orientation: Matrix3::new(1., 0., 0., 0., 1., 0., 0., 0., 1.),
            orientation_rate: Matrix3::<f32>::zero(),
            moi_body_inv: Matrix3::<f32>::zero(),
            moi_global_inv: Matrix3::<f32>::zero(),
            total_mass: 0.,
            nnps_idx: 0,
        }
    }

    pub fn from_xyr(x: Vec<f32>, y: Vec<f32>, rad: Vec<f32>) -> RB3d {
        RB3d::from_xyzr(x, y, vec![0.; rad.len()], rad)
    }
}

impl WriteOutput for RB3d {
    fn write_vtk(&self, output: String) {
        let x = &self.x;
        let y = &self.y;
        let z = &self.z;
        let u = &self.u;
        let v = &self.v;
        let w = &self.w;
        let fx = &self.fx;
        let fy = &self.fy;
        let fz = &self.fz;

        let mut pos = vec![];
        let mut vel = vec![];
        let mut forces = vec![];
        let mut h = vec![];
        let mut m = vec![];
        for i in 0..x.len() {
            pos.push(x[i]);
            pos.push(y[i]);
            pos.push(z[i]);
            vel.push(u[i]);
            vel.push(v[i]);
            vel.push(w[i]);
            forces.push(fx[i]);
            forces.push(fy[i]);
            forces.push(fz[i]);
            h.push(self.h[i]);
            m.push(self.m[i]);
        }

        let mut attributes = Attributes::new();
        attributes.point.push((
            "Forces".to_string(),
            Attribute::Vectors {
                data: forces.into(),
            },
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
            "m".to_string(),
            Attribute::Scalars {
                num_comp: 1,
                lookup_table: None,
                data: m.into(),
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
impl_GetXYZH!(RB3d);

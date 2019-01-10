pub mod equations;

use cgmath::{Matrix3, Vector3};
use cgmath::prelude::*;

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
    pub cm: Vec<Vector3<f32>>,
    pub lin_vel: Vec<Vector3<f32>>,
    pub ang_vel: Vec<Vector3<f32>>,
    pub lin_mom: Vec<Vector3<f32>>,
    pub ang_mom: Vec<Vector3<f32>>,
    pub cm_0: Vec<Vector3<f32>>,
    pub lin_vel_0: Vec<Vector3<f32>>,
    pub ang_mom_0: Vec<Vector3<f32>>,
    pub torque: Vec<Vector3<f32>>,
    pub net_force: Vec<Vector3<f32>>,
    pub orientation: Vec<Matrix3<f32>>,
    pub orientation_0: Vec<Matrix3<f32>>,
    pub orientation_rate: Vec<Matrix3<f32>>,
    pub moi_body_inv: Vec<Matrix3<f32>>,
    pub moi_global_inv: Vec<Matrix3<f32>>,
    pub total_mass: Vec<f32>,
    pub nnps_idx: usize,
    pub body_id: Vec<usize>,
    pub no_bodies: usize,
    pub body_limits: Vec<usize>,
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
            cm: vec![Vector3::new(0., 0., 0.)],
            lin_vel: vec![Vector3::new(0., 0., 0.)],
            ang_vel: vec![Vector3::new(0., 0., 0.)],
            lin_mom: vec![Vector3::new(0., 0., 0.)],
            ang_mom: vec![Vector3::new(0., 0., 0.)],
            cm_0: vec![Vector3::new(0., 0., 0.)],
            lin_vel_0: vec![Vector3::new(0., 0., 0.)],
            ang_mom_0: vec![Vector3::new(0., 0., 0.)],
            torque: vec![Vector3::new(0., 0., 0.)],
            net_force: vec![Vector3::new(0., 0., 0.)],
            orientation: vec![Matrix3::new(1., 0., 0., 0., 1., 0., 0., 0., 1.)],
            orientation_0: vec![Matrix3::new(1., 0., 0., 0., 1., 0., 0., 0., 1.)],
            orientation_rate: vec![Matrix3::<f32>::zero()],
            moi_body_inv: vec![Matrix3::<f32>::zero()],
            moi_global_inv: vec![Matrix3::<f32>::zero()],
            total_mass: vec![0.],
            nnps_idx: 0,
            body_id: vec![0; x.len()],
            no_bodies: 1,
            body_limits: vec![0, x.len()],
        }
    }

    /// Create a rigid body from x, y and radius of Rigid body points.
    /// This will assume the body to have a single body, and sets the body id to 1
    /// If you have Many bodies use from_xyzr_b_id method.
    pub fn from_xyr(x: Vec<f32>, y: Vec<f32>, rad: Vec<f32>) -> RB3d {
        RB3d::from_xyzr(x, y, vec![0.; rad.len()], rad)
    }

    fn set_up_bodies(&mut self, body_id: Vec<usize>) {
        // get the total no of bodies form the maximum index of the b_id
        let total_bodies = body_id.iter().max();
        // set the total no of bodies
        self.no_bodies = *total_bodies.unwrap() + 1;
        println!("Failing here", );
        let no_bodies = self.no_bodies;

        // Create the center of mass properties for all other
        // bodies
        self.cm = vec![Vector3::new(0., 0., 0.); no_bodies];
        self.lin_vel = vec![Vector3::new(0., 0., 0.); no_bodies];
        self.ang_vel = vec![Vector3::new(0., 0., 0.); no_bodies];
        self.lin_mom = vec![Vector3::new(0., 0., 0.); no_bodies];
        self.ang_mom = vec![Vector3::new(0., 0., 0.); no_bodies];
        self.cm_0 = vec![Vector3::new(0., 0., 0.); no_bodies];
        self.lin_vel_0 = vec![Vector3::new(0., 0., 0.); no_bodies];
        self.ang_mom_0 = vec![Vector3::new(0., 0., 0.); no_bodies];
        self.torque = vec![Vector3::new(0., 0., 0.); no_bodies];
        self.net_force = vec![Vector3::new(0., 0., 0.); no_bodies];
        self.orientation = vec![Matrix3::new(1., 0., 0., 0., 1., 0., 0., 0., 1.); no_bodies];
        self.orientation_0 = vec![Matrix3::new(1., 0., 0., 0., 1., 0., 0., 0., 1.); no_bodies];
        self.orientation_rate = vec![Matrix3::<f32>::zero(); no_bodies];
        self.moi_body_inv = vec![Matrix3::<f32>::zero(); no_bodies];
        self.moi_global_inv = vec![Matrix3::<f32>::zero(); no_bodies];

        // set the body id
        self.body_id = body_id.clone();

        // set the bodies limit (indices limits), for example say we have a 3
        // bodies with 4, 3, 4, particles , and the indices of body id look like
        // [0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2]
        // so the limits would be
        // [0, 3, 4, 7, 7, 11]

        // if we have many bodies then we can assign the limits using a for loop
        let mut body_limits = vec![];

        if no_bodies > 1{
            for i in 0..no_bodies-1{
                let min_idx = {
                    let mut j = 0;
                    while i != body_id[j] {
                        j += 1;
                    }
                    j
                };
                let max_idx = {
                    let mut j = min_idx;
                    while i >= body_id[j]{
                        j += 1;
                    }
                    j
                };
                body_limits.push(min_idx);
                body_limits.push(max_idx);
            }
            body_limits.push(body_limits[2*(no_bodies-1) - 1]);
            body_limits.push(body_id.len());
        }

        self.body_limits = body_limits;
    }

    pub fn from_xyr_b_id(x: Vec<f32>, y: Vec<f32>, rad: Vec<f32>, body_id:Vec<usize>) -> RB3d {
        let mut body = RB3d::from_xyzr(x, y, vec![0.; rad.len()], rad);
        body.set_up_bodies(body_id);
        body
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
        let mut rad = vec![];
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
            rad.push(self.rad[i]);
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

        attributes.point.push((
            "rad".to_string(),
            Attribute::Scalars {
                num_comp: 1,
                lookup_table: None,
                data: rad.into(),
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

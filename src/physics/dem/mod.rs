// sub modules
pub mod equations;

// local library imports
use WriteOutput;
// import to implement this trait for nnps functionality
use contact_search::GetXYH;

// std library imports
use std::fs::OpenOptions;
use std::io::Write;

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

impl WriteOutput for DEM{
    fn write_vtk(&self, output: String) {
        // This is taken from
        // https://lorensen.github.io/VTKExamples/site/VTKFileFormats/#legacy-file-examples
        let x = &self.x;
        let y = &self.y;
        let r = &self.r;
        let fx = &self.fx;
        let fy = &self.fy;
        let filename = output;

        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(filename)
            .unwrap();

        writeln!(file, "# vtk DataFile Version 3.0").unwrap();
        writeln!(file, "Time some").unwrap();
        writeln!(file, "ASCII\nDATASET UNSTRUCTURED_GRID").unwrap();

        writeln!(file, "POINTS {} float", x.len()).unwrap();
        for i in 0..x.len() {
            writeln!(file, "{:.4} {:.4} 0.0", x[i], y[i]).unwrap();
        }

        writeln!(file, "POINT_DATA {}", x.len()).unwrap();
        writeln!(file, "SCALARS Diameter float 1").unwrap();
        writeln!(file, "LOOKUP_TABLE default").unwrap();
        for i in 0..x.len() {
            writeln!(file, "{:.4}", r[i]).unwrap();
        }

        writeln!(file, "VECTORS Force float").unwrap();
        for i in 0..x.len() {
            writeln!(file, "{:.4} {:.4} 0.0000", fx[i], fy[i]).unwrap();
        }
    }
}

// implement nnps macro
impl_GetXYH!(DEM);

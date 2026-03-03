use indicatif::{ProgressBar, ProgressStyle};
use rand::Rng;
use prestige::*;
use std::f64::consts::PI;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::sync::Arc;

#[derive(Clone)]
pub struct Options {
    pub layers: usize,
    pub nx: usize,
    pub ny: usize,
    pub radius: f64,
    pub dt: f64,
    pub steps: usize,
    pub backend: String,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            layers: 40,
            nx: 40,
            ny: 40,
            radius: 0.15,
            dt: 1e-4,
            steps: 1000,
            backend: "default".to_string(),
        }
    }
}

pub fn parse_options() -> Options {
    let mut opt = Options::default();

    let mut args = std::env::args().skip(1);

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--layers" => opt.layers = args.next().unwrap().parse().unwrap(),
            "--nx" => opt.nx = args.next().unwrap().parse().unwrap(),
            "--ny" => opt.ny = args.next().unwrap().parse().unwrap(),
            "--radius" => opt.radius = args.next().unwrap().parse().unwrap(),
            "--dt" => opt.dt = args.next().unwrap().parse().unwrap(),
            "--steps" => opt.steps = args.next().unwrap().parse().unwrap(),
            "--backend" => opt.backend = args.next().unwrap(),
            _ => {}
        }
    }

    opt
}

use std::f64;

pub fn build_block_tank(opt: &Options) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let layers = opt.layers;
    let nx = opt.nx;
    let ny = opt.ny;
    let ball_rad = opt.radius;

    let spacing = 2.1 * ball_rad;

    let mut x = Vec::new();
    let mut u = Vec::new();
    let mut m = Vec::new();
    let mut r = Vec::new();
    let mut btype = Vec::new();

    // =====================================================
    // Particle block dimensions
    // =====================================================

    let particle_length = nx as f64 * spacing;
    let particle_height = layers as f64 * spacing;
    let particle_depth = ny as f64 * spacing;

    // =====================================================
    // Tank dimensions
    // =====================================================

    let tank_length = particle_height;
    let tank_height = particle_height + 10.0 * spacing;
    let tank_depth = particle_depth + 20.0 * spacing;

    let Lx = tank_length;
    let Ly = tank_depth;
    let Lz = tank_height;

    // =====================================================
    // 1️⃣ STAGGERED BLOCK
    // =====================================================

    let x_offset = 2. * ball_rad * tank_length;
    let y_offset = spacing;
    let z_offset = spacing;

    for k in 0..layers {
        for j in 0..ny {
            for i in 0..nx {
                let mut shift_x = 0.0;
                let mut shift_y = 0.0;

                if j % 2 == 1 {
                    shift_x = 0.5 * spacing;
                }

                if k % 2 == 1 {
                    shift_y = 0.5 * spacing;
                }

                let xp = x_offset + i as f64 * spacing + shift_x;
                let yp = y_offset + j as f64 * spacing + shift_y;
                let zp = z_offset + k as f64 * spacing;

                if xp < Lx - spacing && yp < Ly - spacing {
                    x.extend_from_slice(&[xp, yp, zp]);
                    u.extend_from_slice(&[0.0, 0.0, 0.0]);
                    m.push(1.0);
                    r.push(ball_rad);
                    btype.push(1.0); // dynamic
                }
            }
        }
    }

    // =====================================================
    // 2️⃣ BOTTOM
    // =====================================================

    let nx_wall = (Lx / spacing) as usize;
    let ny_wall = (Ly / spacing) as usize;

    for j in 0..=ny_wall {
        for i in 0..=nx_wall {
            let xp = i as f64 * spacing;
            let yp = j as f64 * spacing;
            let zp = 0.0;

            x.extend_from_slice(&[xp, yp, zp]);
            u.extend_from_slice(&[0.0, 0.0, 0.0]);
            m.push(0.0);
            r.push(ball_rad);
            btype.push(0.0); // boundary
        }
    }

    // =====================================================
    // 3️⃣ LEFT & RIGHT WALLS
    // =====================================================

    let nz_wall = (Lz / spacing) as usize;

    for k in 0..=nz_wall {
        for j in 0..=ny_wall {
            let yp = j as f64 * spacing;
            let zp = k as f64 * spacing;

            // Left
            x.extend_from_slice(&[0.0, yp, zp]);
            u.extend_from_slice(&[0.0, 0.0, 0.0]);
            m.push(0.0);
            r.push(ball_rad);
            btype.push(0.0);

            // Right
            x.extend_from_slice(&[Lx, yp, zp]);
            u.extend_from_slice(&[0.0, 0.0, 0.0]);
            m.push(0.0);
            r.push(ball_rad);
            btype.push(0.0);
        }
    }

    // =====================================================
    // 4️⃣ FRONT & BACK WALLS
    // =====================================================

    for k in 0..=nz_wall {
        for i in 0..=nx_wall {
            let xp = i as f64 * spacing;
            let zp = k as f64 * spacing;

            // Front
            x.extend_from_slice(&[xp, 0.0, zp]);
            u.extend_from_slice(&[0.0, 0.0, 0.0]);
            m.push(0.0);
            r.push(ball_rad);
            btype.push(0.0);

            // Back
            x.extend_from_slice(&[xp, Ly, zp]);
            u.extend_from_slice(&[0.0, 0.0, 0.0]);
            m.push(0.0);
            r.push(ball_rad);
            btype.push(0.0);
        }
    }

    for i in 0.. m.len(){
        if btype[i] == 1. {
        x[3 * i + 0]  -= 5. * ball_rad;
        x[3 * i + 1]  += 5. * ball_rad;
        x[3 * i + 2]  += 2. * ball_rad;
        }
    }

    (x, u, m, r, btype)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // -----------------------------
    // 1. CUDA init
    // -----------------------------
    let cuda = CudaManager::new(0)?;
    let stream = cuda.new_stream()?;

    // -----------------------------
    // 2. Build tank + particles
    // -----------------------------
    let opt = parse_options();

    let (x, u, m, r, btype) = build_block_tank(&opt);

    let n = (x.len() / 3) as u32;
    let mut particles = Particles::new(n, stream.clone())?;

    stream.memcpy_htod(&x, &mut particles.x)?;
    stream.memcpy_htod(&u, &mut particles.u)?;
    stream.memcpy_htod(&m, &mut particles.m)?;
    stream.memcpy_htod(&r, &mut particles.rad)?;
    stream.memcpy_htod(&btype, &mut particles.body_type)?;

    // -----------------------------
    // 3. Load kernels
    // -----------------------------
    let module = cuda.load_module("cuda_kernels/forces.ptx")?;
    let module_i = cuda.load_module("cuda_kernels/integrate.ptx")?;

    let k_reset = module.get("reset_force")?;
    let k_gravity = module.get("gravity_force")?;
    // let k_dem = module.get("dem_full_allpairs")?;
    let k_dem = module.get("dem_full_neighbors")?;
    let k_freeze_bdry = module.get("freeze_boundary_particles")?;
    let k_stage1 = module_i.get("dem_stage1")?;
    let k_stage2 = module_i.get("dem_stage2")?;
    let k_stage3 = module_i.get("dem_stage3")?;

    // -----------------------------
    // 4. Forces
    // -----------------------------
    let kn = 1e5;
    let cor = 0.1;
    let mu = 0.3;

    let mut reset = ResetForce::new(&particles, k_reset)?;
    let mut gravity = GravityForce::new([0., 0., -9.81], &particles, k_gravity)?;
    let mut dem = DEMParticleParticleForce::new(kn, cor, mu, &particles, k_dem)?;
    let mut freeze_bdry = FreezeBdryForce::new(&particles, k_freeze_bdry)?;

    // -----------------------------
    // 5. Integrator
    // -----------------------------
    let dt = 5e-5;
    let tf = 1.0;
    let steps = (tf / dt) as usize;

    let mut integ = LeapfrogIntegrator::new(dt, &particles, k_stage1, k_stage2, k_stage3)?;
    // -----------------------------
    // 6. Create grid from particle bounds
    // -----------------------------

    // Copy positions to host
    let x_host = particles.stream.clone_dtoh(&particles.x)?;

    // We assume uniform radius for now
    let r_host = particles.stream.clone_dtoh(&particles.rad)?;

    // -------------------------------------------------
    // Compute min / max
    // -------------------------------------------------

    let mut min_x = f64::INFINITY;
    let mut min_y = f64::INFINITY;
    let mut min_z = f64::INFINITY;

    let mut max_x = -f64::INFINITY;
    let mut max_y = -f64::INFINITY;
    let mut max_z = -f64::INFINITY;

    let mut max_rad: f64 = 0.0;

    let n = x_host.len() / 3;

    for i in 0..n {
        let xi = x_host[3 * i + 0];
        let yi = x_host[3 * i + 1];
        let zi = x_host[3 * i + 2];

        min_x = min_x.min(xi);
        min_y = min_y.min(yi);
        min_z = min_z.min(zi);

        max_x = max_x.max(xi);
        max_y = max_y.max(yi);
        max_z = max_z.max(zi);

        max_rad = max_rad.max(r_host[i]);
    }

    // -------------------------------------------------
    // Add padding (5 radii safety margin)
    // -------------------------------------------------

    let padding = 5.0 * max_rad;

    let domain_min = Double3 {
        x: min_x - padding,
        y: min_y - padding,
        z: min_z - padding,
    };

    let domain_max = Double3 {
        x: max_x + padding,
        y: max_y + padding,
        z: max_z + padding,
    };

    // -------------------------------------------------
    // Cell size (DEM contact radius scale)
    // -------------------------------------------------

    let cell_size = 4. * max_rad;
    let inv_h = 1.0 / cell_size;

    let cell_num_x = ((domain_max.x - domain_min.x) * inv_h).ceil() as i32;
    let cell_num_y = ((domain_max.y - domain_min.y) * inv_h).ceil() as i32;
    let cell_num_z = ((domain_max.z - domain_min.z) * inv_h).ceil() as i32;

    let ncell = (cell_num_x * cell_num_y * cell_num_z) as usize;
    let np = particles.n_host[0] as usize;

    let mut grid = Grid::new(
        particles.stream.clone(),
        np,
        ncell,
        cell_num_x,
        cell_num_y,
        cell_num_z,
        inv_h,
        domain_min,
        domain_max,
    )?;

    // Build once before loop
    grid.build(&particles)?;

    // -----------------------------
    // 7. Time loop
    // -----------------------------
    println!("Total no of particles are: {:?}", particles.m.len());
    let file = File::create("tank_probe.csv")?;
    let mut writer = BufWriter::new(file);
    writeln!(writer, "time,z")?;

    let probe = (n - 1) as usize * 3 + 2; // last ball, z index

    let pb = ProgressBar::new(steps as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} ({eta})")
            .unwrap()
            .progress_chars("=>-"),
    );

    for step in 0..steps {
        integ.stage1(&mut particles)?;
        integ.stage2(&mut particles)?;

        grid.build(&particles)?;

        reset.compute(&mut particles)?;
        gravity.compute(&mut particles)?;
        dem.compute_with_neigbors(&mut particles, &mut grid)?;
        freeze_bdry.compute(&mut particles)?;

        integ.stage3(&mut particles)?;
        stream.synchronize()?; // important for stability + logging

        if step % 200 == 0 {
            particles.write_vtk(step)?;
        }
        pb.inc(1);
    }

    pb.finish_with_message("Simulation done");
    writer.flush()?;

    Ok(())
}

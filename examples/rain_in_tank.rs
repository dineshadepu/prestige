use indicatif::{ProgressBar, ProgressStyle};
use rand::Rng;
use prestige::*;
use std::f64::consts::PI;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::sync::Arc;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // -----------------------------
    // 1. CUDA init
    // -----------------------------
    let cuda = CudaManager::new(0)?;
    let stream = cuda.new_stream()?;

    // -----------------------------
    // 2. Build tank + particles
    // -----------------------------
    let R = 2.0; // tank radius
    let H = 4.0; // tank height
    let wall_rad = 0.2;
    let spacing = 2.0 * wall_rad * 0.95;

    let mut x = Vec::new();
    let mut u = Vec::new();
    let mut m = Vec::new();
    let mut r = Vec::new();
    let mut btype = Vec::new();

    // ---- circular floor ----
    let nr = (R / spacing) as usize;
    for ir in 0..nr {
        let rxy = ir as f64 * spacing;
        let ntheta = ((2.0 * PI * rxy) / spacing).max(1.0) as usize;
        for i in 0..ntheta {
            let th = i as f64 * 2.0 * PI / ntheta as f64;
            let xf = rxy * th.cos();
            let yf = rxy * th.sin();

            x.extend_from_slice(&[xf, yf, 0.0]);
            u.extend_from_slice(&[0.0, 0.0, 0.0]);
            m.push(0.0); // frozen
            r.push(wall_rad);
            btype.push(0.0);
        }
    }

    // ---- cylindrical wall ----
    let nz = (H / spacing) as usize;
    let ntheta = (2.0 * PI * R / spacing) as usize;

    for k in 0..nz {
        let z = k as f64 * spacing;
        for i in 0..ntheta {
            let th = i as f64 * 2.0 * PI / ntheta as f64;
            let xw = R * th.cos();
            let yw = R * th.sin();

            x.extend_from_slice(&[xw, yw, z]);
            u.extend_from_slice(&[0.0, 0.0, 0.0]);
            m.push(0.0); // frozen
            r.push(wall_rad);
            btype.push(0.0);
        }
    }

    // ---- falling balls (no overlap) ----
    let mut rng = rand::thread_rng();
    let n_ball = 300;
    let ball_rad = 0.15;

    let min_dist = 2.0 * ball_rad * 1.05;
    let z_min = H - 2.0 * ball_rad;
    let z_max = H + 1.0;

    let mut centers: Vec<(f64, f64, f64)> = Vec::new();

    while centers.len() < n_ball {
        let rxy = rng.r#gen::<f64>().sqrt() * (R - 2.0 * ball_rad);
        let th = rng.r#gen::<f64>() * 2.0 * PI;

        let xb = rxy * th.cos();
        let yb = rxy * th.sin();
        let zb = z_min + (z_max - z_min) * rng.r#gen::<f64>();

        let mut ok = true;
        for &(xj, yj, zj) in &centers {
            let dx = xb - xj;
            let dy = yb - yj;
            let dz = zb - zj;
            if dx * dx + dy * dy + dz * dz < min_dist * min_dist {
                ok = false;
                break;
            }
        }

        if ok {
            centers.push((xb, yb, zb));
            x.extend_from_slice(&[xb, yb, zb]);
            u.extend_from_slice(&[0.0, 0.0, 0.0]);
            m.push(1.0);
            r.push(ball_rad);
            btype.push(1.0);
        }
    }

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
    let kn = 1e4;
    let cor = 0.4;
    let mu = 0.0;

    let mut reset = ResetForce::new(&particles, k_reset)?;
    let mut gravity = GravityForce::new([0., 0., -9.81], &particles, k_gravity)?;
    let mut dem = DEMParticleParticleForce::new(kn, cor, mu, &particles, k_dem)?;
    let mut freeze_bdry = FreezeBdryForce::new(&particles, k_freeze_bdry)?;

    // -----------------------------
    // 5. Integrator
    // -----------------------------
    let dt = 1e-4;
    let tf = 2.0;
    let steps = (tf / dt) as usize;

    let mut integ = LeapfrogIntegrator::new(dt, &particles, k_stage1, k_stage2, k_stage3)?;

    // -----------------------------
    // 6. Create the grid
    // -----------------------------
    let cell_size = wall_rad * 2.;
    let inv_h = 1.0 / (2. * cell_size);

    let domain_min = Double3 {
        x: -H,
        y: -H,
        z: -H,
    };
    let domain_max = Double3 {
        x: H,
        y: H,
        z: H,
    };

    let cell_num_x = ((domain_max.x - domain_min.x) * inv_h) as i32;
    let cell_num_y = ((domain_max.y - domain_min.y) * inv_h) as i32;
    let cell_num_z = ((domain_max.z - domain_min.z) * inv_h) as i32;

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

    // build once before loop
    grid.build(&particles)?;

    // -----------------------------
    // 7. Time loop
    // -----------------------------
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

        if step % 50 == 0 {
            particles.write_vtk(step)?;
        }
        pb.inc(1);
    }

    pb.finish_with_message("Simulation done");
    writer.flush()?;

    Ok(())
}

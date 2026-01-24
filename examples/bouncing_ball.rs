use gpu_particles_no_neighbours::*;
use indicatif::{ProgressBar, ProgressStyle};
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
    // 2. Particles
    // -----------------------------
    let n: u32 = 2;
    let mut particles = Particles::new(n, stream.clone())?;

    // particle 0 = wall at x = 0
    // particle 1 = ball above it
    let x0: Vec<f64> = vec![
        0.0, 0.0, 0.0, // wall
        0.0, 1.0, 0.0, // ball
    ];

    // ball falls down
    let u0: Vec<f64> = vec![
        0.0, 0.0, 0.0, // wall
        0.0, 0.0, 0.0, // ball
    ];

    let m0: Vec<f64> = vec![1e30, 1.0]; // wall is "infinite mass"
    let r0: Vec<f64> = vec![0.5, 0.2]; // wall thickness, ball radius

    // body type: 0 = boundary, 1 = dynamic
    let btype: Vec<f64> = vec![0., 1.];

    stream.memcpy_htod(&x0, &mut particles.x)?;
    stream.memcpy_htod(&u0, &mut particles.u)?;
    stream.memcpy_htod(&m0, &mut particles.m)?;
    stream.memcpy_htod(&r0, &mut particles.rad)?;
    stream.memcpy_htod(&btype, &mut particles.body_type)?;

    // -----------------------------
    // 3. Load kernels
    // -----------------------------
    let module = cuda.load_module("cuda_kernels/forces.ptx")?;
    let module_i = cuda.load_module("cuda_kernels/integrate.ptx")?;

    let k_reset = module.get("reset_force")?;
    let k_gravity = module.get("gravity_force")?;
    let k_dem = module.get("dem_full_allpairs")?;
    let k_freeze_bdry = module.get("freeze_boundary_particles")?;
    let k_stage1 = module_i.get("dem_stage1")?;
    let k_stage2 = module_i.get("dem_stage2")?;
    let k_stage3 = module_i.get("dem_stage3")?;

    // -----------------------------
    // 4. Forces
    // -----------------------------
    let kn = 1e5;
    let cor = 0.7;
    let mu = 0.0;

    let mut reset = ResetForce::new(&particles, k_reset)?;
    let mut gravity = GravityForce::new([0., -9.81, 0.], &particles, k_gravity)?;
    let mut dem = DEMParticleParticleForce::new(kn, cor, mu, &particles, k_dem)?;
    let mut freeze_bdry = FreezeBdryForce::new(&particles, k_freeze_bdry)?;

    // -----------------------------
    // 5. Integrator
    // -----------------------------
    let dt = 1e-5;
    let tf = 2.0;
    let steps = (tf / dt) as usize;

    let mut integ = LeapfrogIntegrator::new(dt, &particles, k_stage1, k_stage2, k_stage3)?;

    // -----------------------------
    // 6. Time loop
    // -----------------------------
    let file = File::create("ball_height.csv")?;
    let mut writer = BufWriter::new(file);
    writeln!(writer, "time,y")?;

    let pb = ProgressBar::new(steps as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} ({eta})")
            .unwrap()
            .progress_chars("=>-"),
    );

    for step in 0..steps {
        // v(n+1/2)
        integ.stage1(&mut particles)?;

        // x(n+1)
        integ.stage2(&mut particles)?;

        // clear forces
        reset.compute(&mut particles)?;
        // Apply gravity
        gravity.compute(&mut particles)?;
        // compute contact forces
        dem.compute(&mut particles)?;
        // Freeze the boundary particles
        freeze_bdry.compute(&mut particles)?;

        // v(n+1)
        integ.stage3(&mut particles)?;

        if step % 200 == 0 {
            // let x_host = stream.clone_dtoh(&particles.x)?;
            // println!("{step:6}  x_1 = {:?}, x_4 = {:?}", x_host[1], x_host[4]);
            let x_host = stream.clone_dtoh(&particles.x)?;
            let y_ball = x_host[4]; // particle 1, y = index 4
            let time = step as f64 * dt;
            writeln!(writer, "{},{}", time, y_ball)?;
        }
        pb.inc(1);
    }
    pb.finish_with_message("Simulation done");
    writer.flush()?;

    Ok(())
}

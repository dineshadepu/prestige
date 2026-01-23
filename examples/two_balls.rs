use gpu_particles::*;
use std::sync::Arc;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cuda = CudaManager::new(0)?;
    let stream = cuda.new_stream()?;

    let n = 2;
    let mut particles = Particles::new(n, stream.clone())?;

    // initial conditions
    let x0 = vec![-1.0, 1.0];
    let vx0 = vec![1.0, -1.0];

    stream.htod_copy_into(&x0, &mut particles.x)?;
    stream.htod_copy_into(&vx0, &mut particles.vx)?;

    // Get all kernels or functions on GPU
    let module = cuda.load_module("cuda_kernels/kernels.ptx")?;
    let kernel_make_forces_zero = module.get("DEM_make_forces_zero")?;
    let kernel_dem = module.get("DEM_force")?;
    let kernel_stage1 = module.get("stage1")?;
    let kernel_stage2 = module.get("stage2")?;
    let kernel_stage3 = module.get("stage3")?;

    let kn = 1e5;
    let dem = DEMParticleParticleForce::new(kn, kernel_dem);

    let dt = 1e-4;
    let integ = LeapfrogIntegrator::new(dt, kernel_stage1, kernel_stage2, kernel_stage3);

    for step in 0..10000 {
        integ.stage1(&mut particles)?;
        integ.stage2(&mut particles)?;

        dem.make_forces_zero(&mut particles)?;
        dem.compute(&mut particles)?;

        integ.stage3(&mut particles)?;

        if step % 100 == 0 {
            let (x, _) = particles.download_positions()?;
            println!("{step} {:?}", x);
        }
    }

    Ok(())
}

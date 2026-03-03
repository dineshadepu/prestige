// Copyright (c) 2026 Dinesh Adepu
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
use prestige::*;
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
    let n = 2;
    let mut particles = Particles::new(n, stream.clone())?;

    // positions: (-1,0,0) and (1,0,0)
    let x0 = vec![-1.0, 0.0, 0.0, 1.0, 0.0, 0.0];

    // velocities: toward each other
    let u0 = vec![1.0, 0.0, 0.0, -1.0, 0.0, 0.0];

    let m0 = vec![1.0, 1.0];
    let r0 = vec![0.5, 0.5];

    stream.memcpy_htod(&x0, &mut particles.x)?;
    stream.memcpy_htod(&u0, &mut particles.u)?;
    stream.memcpy_htod(&m0, &mut particles.m)?;
    stream.memcpy_htod(&r0, &mut particles.rad)?;

    // -----------------------------
    // 3. Load kernels
    // -----------------------------
    let module = cuda.load_module("cuda_kernels/forces.ptx")?;
    let module_i = cuda.load_module("cuda_kernels/integrate.ptx")?;

    let k_reset = module.get("reset_force")?;
    let k_gravity = module.get("gravity_force")?;
    let k_dem = module.get("dem_full_allpairs")?;
    let k_stage1 = module_i.get("dem_stage1")?;
    let k_stage2 = module_i.get("dem_stage2")?;
    let k_stage3 = module_i.get("dem_stage3")?;

    // -----------------------------
    // 4. Forces
    // -----------------------------
    let kn = 1e4;
    let cor = 1.0;
    let mu = 0.0;

    let mut reset = ResetForce::new(&particles, k_reset)?;
    let mut gravity = GravityForce::new([0., 0., 0.], &particles, k_gravity)?;
    let mut dem = DEMParticleParticleForce::new(kn, cor, mu, &particles, k_dem)?;

    // -----------------------------
    // 5. Integrator
    // -----------------------------
    let dt = 1e-4;
    let tf = 2.0;
    let steps = (tf / dt) as usize;

    let mut integ = LeapfrogIntegrator::new(dt, &particles, k_stage1, k_stage2, k_stage3)?;

    // -----------------------------
    // 6. Time loop
    // -----------------------------
    for step in 0..steps {
        // v(n+1/2)
        integ.stage1(&mut particles)?;

        // x(n+1)
        integ.stage2(&mut particles)?;

        // clear forces
        reset.compute(&mut particles);

        // compute contact forces
        dem.compute(&mut particles)?;

        // v(n+1)
        integ.stage3(&mut particles)?;

        if step % 200 == 0 {
            let x_host = stream.clone_dtoh(&particles.x)?;
            println!("{step:6}  x_0 = {:?}, x_1 = {:?}", x_host[0], x_host[3]);
        }
    }

    Ok(())
}

use prestige::*;
use clap::Parser;


class ParticlesDEM:
    def __init__(self, points, max_no_tangential_contacts_limit=6, dim=3):
        # Properties
        self.x = wp.array(points, dtype=wp.vec3)  # Positions
        self.u = wp.zeros(len(points), dtype=wp.vec3)  # Velocities
        self.au = wp.zeros(len(points), dtype=wp.vec3)  # Accelerations
        self.force = wp.zeros(len(points), dtype=wp.vec3)  # Forces
        self.torque = wp.zeros(len(points), dtype=wp.vec3)  # Torques
        self.omega = wp.zeros(len(points), dtype=wp.vec3)  # Angular velocities
        self.m = wp.ones(len(points), dtype=wp.float32)  # Masses
        self.rho = wp.ones(len(points), dtype=wp.float32)  # Densities
        self.rad = wp.ones(len(points), dtype=wp.float32)  # Radii
        self.E = wp.ones(len(points), dtype=wp.float32)  # Young's Modulus
        self.nu = wp.ones(len(points), dtype=wp.float32)  # Poisson's Ratio
        self.G = wp.ones(len(points), dtype=wp.float32)  # Shear Modulus
        self.I = wp.ones(len(points), dtype=wp.float32)  # Moment of inertia

        self.tangential_disp_ss_x = wp.zeros(len(points) * max_no_tangential_contacts_limit, dtype=wp.float32)  # tangential disp in x direction with sphere contact
        self.tangential_disp_ss_y = wp.zeros(len(points) * max_no_tangential_contacts_limit, dtype=wp.float32)  # tangential disp in y direction with sphere contact
        self.tangential_disp_ss_z = wp.zeros(len(points) * max_no_tangential_contacts_limit, dtype=wp.float32)  # tangential disp in z direction with sphere contact
        self.tangential_disp_ss_idx = wp.array(-1 * np.ones(len(points) * max_no_tangential_contacts_limit), dtype=wp.int32)  # tangential disp id with sphere contact

        self.total_no_tangential_contacts_ss = wp.zeros(len(points), dtype=wp.int32)  # total no of tangential contacts
        self.max_no_tangential_contacts_limit_ss = wp.array(np.array([max_no_tangential_contacts_limit]), dtype=wp.int32)  # maximum no of contacts

        # Properties related to contact with infinite wall
        self.tangential_disp_sw_x = wp.zeros(len(points) * 6, dtype=wp.float32)
        self.tangential_disp_sw_y = wp.zeros(len(points) * 6, dtype=wp.float32)
        self.tangential_disp_sw_z = wp.zeros(len(points) * 6, dtype=wp.float32)
        # Properties related to contact with infinite wall ends

        # Grid for neighbours
        nx = int(len(points)**0.5) + 1
        ny = int(len(points)**0.5) + 1
        nz = 1
        if dim == 3:
            nx = int(len(points)**1./3.) + 1
            ny = int(len(points)**1./3.) + 1
            nz = int(len(points)**1./3.) + 1

        # print("nx, ny, nz are", nx, ny, nz)
        self.grid = wp.HashGrid(nx, ny, nz)
        self.max_radius = 0.0

    def output(self, folder, step, time):
        positions = self.x.numpy()
        velocities = self.u.numpy()
        forces = self.force.numpy()
        torques = self.torque.numpy()
        m = self.m.numpy()
        rho = self.rho.numpy()
        rad = self.rad.numpy()

        write_time_step(
            folder+"/particles",       # prefix
            step,                 # timestep index
            time,               # simulation time
            positions,
            ("velocity", velocities),
            ("force", forces),
            ("torque", torques),
            ("mass", m),
            ("density", rho),
            ("radius", rad),
        )

fn run_benchmark0(){
    // Create particles

    // Create wall if needed

    // Create wall if needed

}


#[derive(Parser, Debug)]
#[command(name = "Simulation Selector")]
#[command(about = "Run one of the simulation examples", long_about = None)]
pub struct Cli {
    /// Example number to run (0 to 7)
    #[arg(short, long)]
    pub example_id: usize,
}

fn main() {
    let cli = Cli::parse();

    match cli.example_id {
        0 => run_example0(),
        1 => run_example1(),
        2 => run_example2(),
        3 => run_example3(),
        4 => run_example4(),
        5 => run_example5(),
        6 => run_example6(),
        7 => run_example7(),
        _ => {
            eprintln!("Invalid example_id: must be between 0 and 7");
            std::process::exit(1);
        }
    }
}


fn main() {

    let dim = 3;

    let x_sand = vec![1.0, 2.0];
    let y_sand = vec![1.0, 1.0];
    let z_sand = vec![0.0, 0.0];
    let neighbour_radius_sand = vec![0.2, 0.2];
    let rho_sand = vec![1000.0, 1000.0];
    let radius_sand = vec![0.1, 0.1];
    let dem_id_sand = vec![0, 0];

    let mut sand = ParticlesDEM::new(dim, "sand".to_string(), &x_sand, &y_sand, &z_sand, &neighbour_radius_sand, &rho_sand, &radius_sand, &dem_id_sand);

    let x_body = vec![1.0, 2.0];
    let y_body = vec![-0.5, 0.5];
    let z_body = vec![0.0, 0.0];
    let neighbour_radius_body = vec![0.2, 0.2];
    let rho_body = vec![1000.0, 1000.0];
    let radius_body = vec![0.1, 0.1];
    let dem_id_body = vec![1, 1];
    let mut body = ParticlesDEM::new(dim, "body".to_string(), &x_body, &y_body, &z_body, &neighbour_radius_body, &rho_body, &radius_body, &dem_id_body);

    let x_boundary = vec![1.0, 2.0];
    let y_boundary = vec![-0.5, 0.5];
    let z_boundary = vec![0.0, 0.0];
    let neighbour_radius_boundary = vec![0.2, 0.2];
    let rho_boundary = vec![1000.0, 1000.0];
    let radius_boundary = vec![0.1, 0.1];
    let dem_id_boundary = vec![2, 2];
    let mut boundary = ParticlesDEM::new(dim, "boundary".to_string(), &x_boundary, &y_boundary, &z_boundary, &neighbour_radius_boundary, &rho_boundary, &radius_boundary, &dem_id_boundary);

    let dt = 1e-4;
    let tf = 1.;
    let num_steps = (tf / dt) as usize;
    let mut t = 0.;
    for step in 0..=num_steps {
        // loop body
        sand.stage1(dt);

        sand.stage2(dt);

        // update the neighbours
        // neighbors->build( x, 0, x.size(), delta,
        //                   cell_ratio, mesh_min, mesh_max );

        apply_hertz_contact_force_DEM_macro!(sand, (sand, body, boundary), t, dt, 0.2, 0.3);
        update_dem_tng_pp_contacts_macro!(sand, (sand, body, boundary), t, dt);

        // println!("Updated fx: {:?}", sand.get_fx());


        sand.stage3(dt);

        // sand.output_particles( step );
        // sand.output_data(t);
        t = t + dt;
    }
}

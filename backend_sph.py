import numpy as np
import warp as wp
from hashgrid import HashgridGPU
from kernel_func import cubicKernel, cubicKernelDerivative

DEVICE = "cuda"

RHO0 = 1000.0
C0 = 20.0  # speed of sound
GAMMA_VAL = 7.0
B = RHO0 * (C0**2) / GAMMA_VAL  

dx = 0.2
df_frac = 1.4
H = dx * df_frac

# initial dt
CFL_FACTOR = 0.1
dt_init = CFL_FACTOR * H / C0 

MU = 0.2

# we use the area times density as the mass of each particle
MASS = (dx**2.0) * RHO0


@wp.kernel
def compute_pressure(
    densities: wp.array(dtype=float),
    B: float,
    rho0: float,
    gamma_val: float,
    pressures: wp.array(dtype=float)
):
    i = wp.tid()
    rho_i = densities[i]
    # tait equation of state
    pressures[i] = B * ((rho_i / rho0) ** gamma_val - 1.0)


@wp.kernel
def compute_acceleration(
    positions: wp.array(dtype=wp.vec2),
    velocities: wp.array(dtype=wp.vec2),
    densities: wp.array(dtype=float),
    pressures: wp.array(dtype=wp.float32),
    neighbors: wp.array2d(dtype=wp.int32),
    neighbor_counts: wp.array(dtype=wp.int32),
    d_rho: wp.array(dtype=wp.float32),
    mass_val: float,
    h: float,
    c: float,
    mu: float,
    gravity: wp.vec2,
    accel_out: wp.array(dtype=wp.vec2)
):
    i = wp.tid()
    pos_i = positions[i]
    vel_i = velocities[i]
    press_i = pressures[i]
    dens_i = densities[i]
    count_i = neighbor_counts[i]
    a_press = wp.vec2(0.0, 0.0)
    d_rho_i = wp.float32(0.0)
    a_visc = wp.vec2(0.0, 0.0)
    for n in range(count_i):
        j = neighbors[i, n] # we need to add 1 to skip the first element which is the count
        if j == -1: # no more neighbors
            break
        if j == i: # skip the particle itself
            continue
        pos_j = positions[j]
        vel_j = velocities[j]
        press_j = pressures[j]
        dens_j = densities[j]
        r_vec = pos_i - pos_j
        r_mod = wp.length(r_vec)
        grad_w = cubicKernelDerivative(r_mod, h)
        # density change rate
        d_rho_i += mass_val * grad_w * wp.dot(vel_i - vel_j, r_vec/r_mod) 
        # pressure term
        a_press += -mass_val * (press_i/(dens_i**2.0+1e-5) + press_j/(dens_j**2.0+1e-5)) * grad_w * r_vec / r_mod
        # viscosity term
        v_ij = vel_i - vel_j
        r = wp.length(r_vec)
        if r > 1e-8 and wp.dot(v_ij, r_vec) < 0.0:
            v_mu = -2.0 * mu * dx * c / (dens_i + dens_j)
            a_visc += -mass_val * v_mu * wp.dot(v_ij, r_vec)/ (r_mod**2.0 + 0.01 * dx**2.0) * grad_w * r_vec / r_mod
            
    accel_out[i] = a_press + a_visc + gravity
    d_rho[i] = d_rho_i
    

@wp.kernel
def semi_implicit_euler(
    positions: wp.array(dtype=wp.vec2),
    velocities: wp.array(dtype=wp.vec2),
    accelerations: wp.array(dtype=wp.vec2),
    densities: wp.array(dtype=wp.float32),
    drho: wp.array(dtype=wp.float32),
    dt: float
):
    i = wp.tid()
    positions[i] = positions[i] + velocities[i] * dt

    velocities[i] = velocities[i] + accelerations[i] * dt
    # positions[i] = positions[i] + velocities[i] * dt
    densities[i] = densities[i] + drho[i] * dt
    
@wp.func
def simulate_collision(positions: wp.array(dtype=wp.vec2), 
                       velocities: wp.array(dtype=wp.vec2), 
                       i:int, 
                       normal: wp.vec2, 
                       penetration: float):
    c_f = 0.7
    positions[i] += normal * (penetration+0.00001)
    velocities[i] -= (1.0 + c_f) * wp.dot(velocities[i], normal) * normal
    
@wp.kernel
def handle_boundaries(    
        positions: wp.array(dtype=wp.vec2),
        velocities: wp.array(dtype=wp.vec2),
        box_min: wp.vec2,
        box_max: wp.vec2):
    
    i = wp.tid()
    pos = positions[i]
    vel = velocities[i]
    if pos[0] < box_min[0]:
        simulate_collision(positions, velocities, i, wp.vec2(1.0, 0.0), box_min[0]-pos[0])
    if pos[0] > box_max[0]:
        simulate_collision(positions, velocities, i, wp.vec2(-1.0, 0.0), pos[0]-box_max[0])
    if pos[1] < box_min[1]:
        simulate_collision(positions, velocities, i, wp.vec2(0.0, 1.0), box_min[1]-pos[1])
    if pos[1] > box_max[1]:
        simulate_collision(positions, velocities, i, wp.vec2(0.0, -1.0), pos[1]-box_max[1])


class SPHSimulation2D_Warp:
    def __init__(self,
                 num_particles=400,
                 box_min=np.array([0.0, 0.0]),
                 box_max=np.array([20.0, 10.0]),
                 mass=MASS,
                 dt=dt_init,
                 gravity=np.array([0.0, -9.8]),
                 cell_size=2 * H):
        self.num_particles = num_particles
        self.box_min = box_min
        self.box_max = box_max
        self.mass = mass
        self.dt = dt
        self.gravity = gravity
        self.cell_size = cell_size
        
        self.positions_cpu = self._init_particles_in_center(num_particles, box_min, box_max)
        self.velocities_cpu = np.zeros((num_particles, 2), dtype=np.float32)
        self.densities_cpu = np.ones(num_particles, dtype=np.float32)*1000.0
        self.pressures_cpu = np.zeros(num_particles, dtype=np.float32)

        self.positions_wp = wp.array(self.positions_cpu, dtype=wp.vec2, device=DEVICE)
        self.velocities_wp = wp.zeros(num_particles, dtype=wp.vec2, device=DEVICE)
        self.densities_wp = wp.ones(num_particles, dtype=wp.float32, device=DEVICE)
        self.densities_wp.fill_(RHO0)
        self.pressures_wp = wp.zeros(num_particles, dtype=wp.float32, device=DEVICE)
        self.accelerations_wp = wp.zeros(num_particles, dtype=wp.vec2, device=DEVICE)
        self.d_rho = wp.zeros(num_particles, dtype=wp.float32, device=DEVICE)

        self.hash_grid = HashgridGPU(dim_x=int((box_max[0] - box_min[0]) / cell_size+1),
                                dim_y=int((box_max[1] - box_min[1]) / cell_size+1),
                                cell_size=cell_size,
                                num_particles=num_particles,
                                positions=wp.array(self.positions_cpu, dtype=wp.vec2, device=DEVICE))

        self.neighbors_wp = wp.full(shape=(num_particles, 128), # 128 is the maximum number of neighbors
                                    value=-1,
                                    dtype=wp.int32,
                                    device=DEVICE)
        self.neighbor_counts_wp = wp.zeros(num_particles, dtype=wp.int32, device=DEVICE)

        self.h = H
        self.rho0 = RHO0
        self.B = B
        self.gamma_val = GAMMA_VAL
        self.c0 = C0
        self.mu = MU
        self.cfl_factor = CFL_FACTOR

    def _init_particles_in_center(self, num_particles, box_min, box_max):
        grid_size = int(np.sqrt(num_particles))
        x_start = box_min[0] + 0.2 
        y_start = box_min[1] + 0.2
        positions = []
        for i in range(grid_size):
            for j in range(grid_size):
                x = x_start + dx * i + 2.0
                y = y_start + dx * j + 2.0
                positions.append([x, y])
        return np.array(positions, dtype=np.float32)


    def sph_simulation_step(self):
        
        # compute the neighbors 
        self.hash_grid.build(self.positions_wp)
        self.hash_grid.query(self.h)
        self.neighbors_wp = self.hash_grid.neighbors
        self.neighbor_counts_wp = self.hash_grid.num_neighbors
        
        # pressure term
        wp.launch(
            kernel=compute_pressure,
            dim=self.num_particles,
            inputs=[
                self.densities_wp,
                self.B,
                self.rho0,
                self.gamma_val
            ],
            outputs=[self.pressures_wp],
            device=DEVICE
        )
        
        # acceleration term: pressure, viscosity, gravity, as well as density change rate
        wp.launch(
            kernel=compute_acceleration,
            dim=self.num_particles,
            inputs=[
                self.positions_wp,
                self.velocities_wp,
                self.densities_wp,
                self.pressures_wp,
                self.neighbors_wp,
                self.neighbor_counts_wp,
                self.d_rho,
                self.mass,
                self.h,
                self.c0,
                self.mu,
                wp.vec2(self.gravity[0], self.gravity[1])
            ],
            outputs=[self.accelerations_wp],
            device=DEVICE
        )

        # semi-implicit Euler integration
        wp.launch(
            kernel=semi_implicit_euler,
            dim=self.num_particles,
            inputs=[
                self.positions_wp,
                self.velocities_wp,
                self.accelerations_wp,
                self.densities_wp,
                self.d_rho,
                self.dt
            ],
            outputs=[],
            device=DEVICE
        )
        
        # boundary handling
        wp.launch(
            kernel=handle_boundaries,
            dim=self.num_particles,
            inputs=[
                self.positions_wp,
                self.velocities_wp,
                wp.vec2(self.box_min[0], self.box_min[1]),
                wp.vec2(self.box_max[0], self.box_max[1])
            ],
            outputs=[],
            device=DEVICE
        )
        

        # update the CPU arrays, 
        self.positions_cpu = self.positions_wp.numpy()
        self.velocities_cpu = self.velocities_wp.numpy()
        self.densities_cpu = self.densities_wp.numpy()
        self.pressures_cpu = self.pressures_wp.numpy()
        accelerations_cpu = self.accelerations_wp.numpy()

        # update the time step
        max_v = np.max(np.linalg.norm(self.velocities_cpu, axis=1))
        max_a = np.max(np.linalg.norm(accelerations_cpu, axis=1))
        max_rho = np.max(self.densities_cpu)

        # compute the time step
        if max_v < 1e-10:
            dt_cfl = 1e10
        else:
            dt_cfl = self.h / max_v
        if max_a < 1e-10:
            dt_f = 1e10
        else:
            dt_f = np.sqrt(self.h / max_a)
        dt_a = self.h / (self.c0 * np.sqrt((max_rho / self.rho0)**self.gamma_val))

        new_dt = self.cfl_factor * min(dt_cfl, dt_f, dt_a)
        
        # prevent the time step from increasing too fast and down to 0
        self.dt = max(1e-6, min(new_dt, 1.2*self.dt))
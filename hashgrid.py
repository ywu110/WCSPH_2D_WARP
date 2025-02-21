import warp as wp

wp.init()

@wp.kernel
def init_grid_count(grid: wp.array3d(dtype=wp.int32),
                    dim_x: int,
                    dim_y: int):
    tid = wp.tid()
    x = tid // dim_y  
    y = tid % dim_y
    if x < dim_x and y < dim_y:
        grid[x, y, 0] = 0  

class HashgridGPU:
    def __init__(self, dim_x: int,
                       dim_y: int,
                       cell_size: float,
                       num_particles: int,
                       positions: wp.array(dtype=wp.vec2),
                       max_particles_per_cell: int = 128):
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.cell_size = cell_size
        self.num_particles = num_particles
        self.positions = positions
        self.max_particle_per_cell = max_particles_per_cell
        
        # Create the hash grid: We use the first element of the third dimension to store the count
        self.grid = wp.zeros((dim_x, dim_y, max_particles_per_cell), dtype=wp.int32)
        self.grid.fill_(-1)  # 初始化为-1
        
        # set the grid count to 0: grid[x, y, 0] = 0. We use the first element of the third dimension to store the count
        wp.launch(kernel=init_grid_count,
                  dim=dim_x*dim_y,
                  inputs=[self.grid, dim_x, dim_y])
        
        # 
        self.num_neighbors = wp.zeros(num_particles, dtype=wp.int32)
        self.neighbors = wp.zeros((num_particles, max_particles_per_cell), dtype=wp.int32)
        self.neighbors.fill_(-1)

    @wp.kernel
    def build_hash_grid(positions: wp.array(dtype=wp.vec2),
                        grid: wp.array3d(dtype=wp.int32),
                        max_particle_per_cell: int,
                        cell_size: float,
                        dim_x: int,
                        dim_y: int):
        tid = wp.tid()
        pos = positions[tid]
        
        # compute the corresponding index in the grid
        cell_x = int(wp.floor(pos[0] / cell_size))
        cell_y = int(wp.floor(pos[1] / cell_size))
        
        # make sure the index is within the grid
        cell_x = wp.clamp(cell_x, 0, dim_x - 1)
        cell_y = wp.clamp(cell_y, 0, dim_y - 1)
        
        # atomic add to the count. The 0 index is used to store the count. 
        # NOTE: atomic_add returns the old value before the add
        count = wp.atomic_add(grid, cell_x, cell_y, 0, 1)
        
        # store the particle index in the grid
        if count < (max_particle_per_cell - 1):  # the first element is used to store the count, so the max_particle_per_cell - 1
            grid[cell_x, cell_y, count+1] = tid  # count is the old value before the add, so we need to add 1
    
    @wp.kernel
    def find_neighbors(positions: wp.array(dtype=wp.vec2),
                       grid: wp.array3d(dtype=wp.int32),
                       cell_size: float,
                       dim_x: int,
                       dim_y: int,
                       max_particle_per_cell: int,
                       radius: float,
                       num_neighbors: wp.array(dtype=wp.int32),
                       neighbors: wp.array2d(dtype=wp.int32)):
        tid = wp.tid()
        pos = positions[tid]
        
        # compute the corresponding index in the grid
        cell_x = int(wp.floor(pos[0] / cell_size))
        cell_y = int(wp.floor(pos[1] / cell_size))
        cell_x = wp.clamp(cell_x, 0, dim_x - 1)
        cell_y = wp.clamp(cell_y, 0, dim_y - 1)
        
        neighbor_count = wp.int(0)
        
        # search the neighboring cells
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                x = cell_x + dx
                y = cell_y + dy
                
                if x >= 0 and x < dim_x and y >= 0 and y < dim_y:
                    # get the number of particles in the cell
                    count = grid[x, y, 0]
                    max_idx = wp.min(count, max_particle_per_cell - 1)
                    
                    # iterate over the particles in the cell
                    for i in range(max_idx):
                        idx = i + 1  # 数据从索引1开始
                        particle_idx = grid[x, y, idx]
                        
                        # make sure the particle index is valid: here we don't ignore the particle itself
                        if particle_idx != -1:
                            neighbor_pos = positions[particle_idx]
                            dist = wp.length(neighbor_pos - pos)
                            if dist < radius:
                                neighbors[tid, neighbor_count] = particle_idx
                                neighbor_count += 1
                        
        # 记录邻居数量
        num_neighbors[tid] = neighbor_count
    
    def build(self, positions: wp.array(dtype=wp.vec2)):
        self.positions = positions
    
        # Create the hash grid: We use the first element of the third dimension to store the count
        self.grid = wp.zeros((self.dim_x, self.dim_y, self.max_particle_per_cell), dtype=wp.int32)
        self.grid.fill_(-1)  # 初始化为-1
        
        # set the grid count to 0: grid[x, y, 0] = 0. We use the first element of the third dimension to store the count
        wp.launch(kernel=init_grid_count,
                  dim=self.dim_x*self.dim_y,
                  inputs=[self.grid, self.dim_x, self.dim_y])
        
        # 
        self.num_neighbors = wp.zeros(self.num_particles, dtype=wp.int32)
        self.neighbors = wp.zeros((self.num_particles, self.max_particle_per_cell), dtype=wp.int32)
        self.neighbors.fill_(-1)

        # build the hash grid
        wp.launch(kernel=self.build_hash_grid,
                  dim=self.num_particles,
                  inputs=[self.positions, self.grid, self.max_particle_per_cell, self.cell_size, self.dim_x, self.dim_y])
    
    
    def query(self, radius: float):
        wp.launch(kernel=self.find_neighbors,
                  dim=self.num_particles,
                  inputs=[self.positions, self.grid, self.cell_size, self.dim_x, self.dim_y, self.max_particle_per_cell, radius, self.num_neighbors, self.neighbors])
    
        
        
        
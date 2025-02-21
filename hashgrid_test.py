from sklearn.neighbors import NearestNeighbors
import numpy as np
import warp as wp
import time

from hashgrid import HashgridGPU

# 创建一些随机的二维点
points = np.random.rand(10000, 2) * 10

current_time = time.time()
# 创建 NearestNeighbors 对象
nbrs = NearestNeighbors(radius=0.1, algorithm='kd_tree', metric='euclidean')
nbrs.fit(points)

num_neighbors_list = []

current_time = time.time()
for i in range(len(points)):
    query_point = np.array([points[i]])
    distances, indices = nbrs.radius_neighbors(query_point)
    num_neighbors = len(indices[0])
    num_neighbors_list.append(num_neighbors)
print(f"Time elapsed: {time.time() - current_time}")


### The following is for GPU acceleration

# Initialize the points and the hash grid
wp.init()
points_gpu = wp.array(points, dtype=wp.vec2)
grid_gpu = HashgridGPU(dim_x=50, dim_y=50, cell_size=0.2, num_particles=10000, positions=points_gpu)

current_time = time.time()  
grid_gpu.build()
grid_gpu.query(0.1)
print(f"Time elapsed: {time.time() - current_time}")

num_neighbors_gpu_list = grid_gpu.num_neighbors.numpy()

# compare the results
for i in range(len(num_neighbors_list)):
    if num_neighbors_list[i] != num_neighbors_gpu_list[i]:
        print(f"Error at index {i}: {num_neighbors_list[i]} != {num_neighbors_gpu_list[i]}")

assert np.allclose(num_neighbors_list, num_neighbors_gpu_list)
print("All results are correct!")
        
        

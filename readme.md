# 2D WCSPH Fluid Simulation using NVIDIA Warp

## Introduction

This repository implements a 2D Weakly Compressible Smoothed Particle Hydrodynamics (WCSPH) fluid simulation fully accelerated on GPU using [NVIDIA's Warp](https://github.com/NVIDIA/warp) framework. Key features include:

* Full GPU acceleration of SPH computation using Warp
* Optimized neighbor search via Warp-based hash grid
* Adaptive time-stepping with Courant-Friedrichs-Lewy (CFL) condition
* Include a PyQT5 GUI for visualization

## WCSPH Algorithm Overview

The WCSPH algorithm is a variant of the Smoothed Particle Hydrodynamics (SPH) method that is designed to simulate weakly compressible fluids. The key idea is to use a more accurate equation of state (EOS) that accounts for the compressibility of the fluid. The WCSPH algorithm can be summarized as follows:

### Pressure Calculation

```math
p_i = B\left[ \left(\frac{\rho_i}{\rho_0}\right)^\gamma - 1 \right],
```
where: 

* $p_i$ is the pressure of particle $i$.
* $B$ is the bulk modulus of the fluid. In our implementation, we set $B = \frac{\rho_0 c_0^2}{\gamma}$.
* $\rho_i$ is the density of particle $i$, and $\rho_0$ is the reference density. Here, we set $\rho_0 = 1000 \, \text{kg/m}^3$.
* $\gamma$ is the adiabatic index of the fluid. In our implementation, we set $\gamma = 7$ for water.
* $c_0$ is the speed of sound in the fluid. In our implementation, we set $c_0 = 20 \, \text{m/s}$.

### Density Update

In many applications, they use the following density update equation:
$\rho_i = \sum_{j} m_j W_{ij}.$ In our implementation, we use the following density update equation:
$$\dfrac{d \rho_i}{dt} = \sum_j m_j (\mathbf{v_i} - \mathbf{v_j}) \cdot \nabla W_{ij}. $$

In the weakly compressible Smoothed Particle Hydrodynamics (WCSPH) framework, the starting point for deriving the density update equation is the continuity equation in its Lagrangian form:

$$
\frac{d \rho}{dt} = \rho \nabla \cdot \mathbf{v}. \quad\quad \text{(1)}
$$

This equation states that the rate of change of the density $\rho$ for a fluid parcel moving with velocity $\mathbf{v}$ is related to the local divergence of $\mathbf{v}$. In other words, if the velocity field converges (negative divergence), the density increases, and if it diverges (positive divergence), the density decreases.

In SPH, one replaces continuous integrals by discrete summations over neighboring particles. Each particle $j$ has a fixed mass $m_j$ and a density $\rho_j$. The first step is the kernel approximation, which expresses a function $f(\mathbf{r})$ at position $\mathbf{r}_i$ as an integral involving a kernel function $W(\mathbf{r}, h)$. In the discrete version, one writes:

```math
f(\mathbf{r}_i) \approx \sum_{j} \frac{m_j}{\rho_j}\, f(\mathbf{r}_j)\, W(\mathbf{r}_i - \mathbf{r}_j, h).
```

When $f$ is a velocity field $\mathbf{v}$, its divergence $\nabla \cdot \mathbf{v}$ can be approximated by taking the divergence of the above summation. A standard SPH form for the divergence is:

```math
\nabla \cdot \mathbf{v}(\mathbf{r}_i) \approx \sum_{j} \frac{m_j}{\rho_j} (\mathbf{v}_j - \mathbf{v}_i) \cdot \nabla W(\mathbf{r}_i - \mathbf{r}_j, h),
```

where $\nabla W(\mathbf{r}_i - \mathbf{r}_j, h)$ is the gradient of the kernel function with respect to $\mathbf{r}_i$. Although there are several slightly different SPH formulations for gradients and divergences (for example with extra symmetrization factors), the underlying idea is the same: the continuous operator $\nabla \cdot$ is replaced by a summation involving kernel gradients and the velocity differences between neighbors.

Substituting this approximation into equation (1), we obtain:

```math
\frac{d \rho_i}{dt} \approx \rho_i(
\sum_{j}
\frac{m_j}{\rho_j}
(\mathbf{v}_j - \mathbf{v}_i) \cdot\nabla W(\mathbf{r}_i - \mathbf{r}_j, h)).
```

In many practical implementations, one may assume $\rho_i \approx \rho_j$ or perform an additional simplification, which leads to a commonly seen SPH continuity form:

```math
\frac{d \rho_i}{dt} \approx \sum_{j} m_j (\mathbf{v}_j - \mathbf{v}_i) \cdot\nabla W(\mathbf{r}_i - \mathbf{r}_j, h)). \quad \quad (2)
```

This final equation (2) is the form that appears in many WCSPH codes and is likely reflected in the code snippet. It shows how the rate of change of the density of particle $i$ arises from the velocity differences with respect to each neighbor $j$, weighted by the kernel gradient. When integrated in time, this approach yields the evolution of $\rho_i$ based on the local velocity field, thereby satisfying the continuity equation in a discrete sense.

### Acceleration Computation
The total acceleration on particle $i$ arises from pressure forces, viscosity, and external forces such as gravity. We write:

```math
\mathbf{a}_i = \mathbf{a}_{\text{pressure}, i} + \mathbf{a}_{\text{viscosity}, i} + \mathbf{g}.
```

1. **Pressure Force**: The pressure force on particle $i$ is given by:

```math
\mathbf{a}_{\text{pressure}, i} = -\sum_j m_j \left( \frac{p_i}{\rho_i^2} + \frac{p_j}{\rho_j^2} \right) \nabla W(\mathbf{r}_i - \mathbf{r}_j, h).
```

2. **Viscosity Force**: The viscosity force on particle $i$ is given by:
```math
\mathbf{a}_{\text{viscosity}, i} =  \sum_j \mu_{ij} m_j \left( \mathbf{v}_j - \mathbf{v}_i \right) \nabla W(\mathbf{r}_i - \mathbf{r}_j, h),
```
where $\mu_{ij} = \dfrac{\alpha c_0 h }{\rho_i + \rho_j}$ is the viscosity coefficient.

## Warp-based Hash Grid
The neighbor search is accelerated using a **GPU-optimized spatial hash grid (HashGridGPU class)** implemented with Warp primitives. Key optimizations:

* Parallel grid cell construction
* Warp-native atomic operations
* Coalesced memory access patterns
* Zero CPU-GPU data transfer during search

## Requirements
* NVIDIA GPU with CUDA support
* Warp 1.6.0
* Python 3.8+

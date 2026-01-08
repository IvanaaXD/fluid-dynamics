# Fluid Dynamics Simulation - Lattice Boltzmann Method (HPC)

## Project Overview

This project implements a 2D fluid dynamics simulation using the **Lattice Boltzmann Method (LBM)** with a D2Q9 lattice model. The goal is to simulate incompressible flow around an obstacle (e.g., a cylinder) and compare performance across two languages and two execution modes:

* **Python:** Sequential and Parallel (Multiprocessing).

* **Rust:** Sequential and Parallel (Threads).


The project involves strong and weak scaling experiments, Amdahl/Gustafson analysis, and a visualization of the flow field.

## Technical Specification

### Grid Representation

The simulation domain is represented as a 2D grid where each node contains 9 discrete velocity directions (D2Q9 model).

* **Data Structures:** 3D arrays (Height x Width x 9) to store particle populations (distribution functions).

* **Boundary Conditions:** Implementation of "No-slip" (bounce-back) for obstacles and "Zou-He" or simple periodic boundaries for inlet/outlet.
  

### Algorithm

The LBM iteration consists of two main steps:

1. **Collision Step:** Local relaxation of particle distributions toward equilibrium based on the BGK operator.
2. **Streaming Step:** Advection of distributions to neighboring nodes.
3. **Macroscopic Variables:** Calculation of fluid density () and velocity () from the distributions.


### Parallel Strategy

* **Python:** Domain decomposition by splitting the grid into horizontal or vertical strips. Uses `multiprocessing` with shared memory arrays to minimize IPC overhead between iterations.

* **Rust:** Row-block partitioning of the grid using native threads. Each thread processes a sub-grid, with synchronization (barrier) required after the streaming step to ensure data consistency.


### Benchmarking

In accordance with the course requirements:

* **Strong Scaling:** Fixed grid size (e.g., 400x100), varying core count to verify Amdahl’s Law.

* **Weak Scaling:** Increasing grid size proportional to the number of cores to verify Gustafson’s Law.

* **Execution:** Each configuration is executed ~30 times to ensure statistical relevance (mean, std dev, outliers).


### Visualization

A Rust-based tool (using the `plotters` library) to generate heatmaps of velocity magnitude and density from the exported iteration data.

---

**Student:** Ivana Radovanovic

**Index:** SV 23/2022

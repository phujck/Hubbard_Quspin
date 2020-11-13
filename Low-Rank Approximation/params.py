import numpy as np

L = 6 # system size
N_up = L // 2 + L % 2  # number of fermions with spin up
N_down = L // 2  # number of fermions with spin down
# N_down = 0
N = N_up + N_down  # number of particles
t0 = 0.52  # hopping strength
# U = 0*t0  # interaction strength
U = 1 * t0  # interaction strength
gamma = 0.05
mu = 0.9
pbc = False
rank=64
"""Laser pulse parameters"""
field = 32.9  # field angular frequency THz
F0 = 10  # Field amplitude MV/cm
a = 4  # Lattice constant Angstroms

t_max = 100
n_steps=400
times, delta = np.linspace(0.0, t_max, n_steps,endpoint=True, retstep=True)


L2 = 6  # system size
N_up2 = L2 // 2 + L % 2  # number of fermions with spin up
N_down2 = L2 // 2  # number of fermions with spin down
# N_down = 0
N2 = N_up + N_down  # number of particles
t02 = 0.52  # hopping strength
# U = 0*t0  # interaction strength
U2 = 1* t0  # interaction strength
gamma2 = 0
mu2 = 0.5
pbc2 = False
rank2=2

t_max2 = t_max
n_steps2 = n_steps
n_steps2 = 400
# t_max2 = 1e-3

times2, delta2 = np.linspace(0.0, t_max, n_steps2, endpoint=True, retstep=True)
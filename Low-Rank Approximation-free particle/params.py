import numpy as np

L = 12 # system size
N_up = L // 2 + L % 2  # number of fermions with spin up
N_down = L // 2  # number of fermions with spin down
# N_down = 0
N = N_up + N_down  # number of particles
t0 = 0.52  # hopping strength
# U = 0*t0  # interaction strength
U = 1 * t0  # interaction strength
gamma = 0.1*t0
mu = 1
pbc = False
rank=16
"""Laser pulse parameters"""
field = 32.9  # field angular frequency THz
F0 = 10  # Field amplitude MV/cm
a = 4  # Lattice constant Angstroms

t_max = 300
n_steps=600
times, delta = np.linspace(0.0, t_max, n_steps,endpoint=True, retstep=True)


L2 = 14  # system size
N_up2 = L2 // 2 + L % 2  # number of fermions with spin up
N_down2 = L2 // 2  # number of fermions with spin down
# N_down = 0
N2 = N_up + N_down  # number of particles
t02 = 0.52  # hopping strength
# U = 0*t0  # interaction strength
U2 = 1* t0  # interaction strength
gamma2 = 0.5*t0
mu2 = 1
pbc2 = False
rank2=16

t_max2 = t_max
n_steps2 = n_steps
n_steps2 = 600
# t_max2 = 1e-3

times2, delta2 = np.linspace(0.0, t_max, n_steps2, endpoint=True, retstep=True)
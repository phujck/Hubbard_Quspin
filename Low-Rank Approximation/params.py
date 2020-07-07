import numpy as np

L = 4  # system size
N_up = L // 2 + L % 2  # number of fermions with spin up
N_down = L // 2  # number of fermions with spin down
# N_down = 0
N = N_up + N_down  # number of particles
t0 = 0.52  # hopping strength
# U = 0*t0  # interaction strength
U = 1 * t0  # interaction strength
gamma = 0.01
mu = 0.11
pbc = False

"""Laser pulse parameters"""
field = 32.9  # field angular frequency THz
F0 = 10  # Field amplitude MV/cm
a = 4  # Lattice constant Angstroms



"""This is used for setting up Hamiltonian in Quspin.  We'll stick with a static Hamiltonian here"""

"""System Evolution Time"""
t_max = 50
n_steps=500
times, delta = np.linspace(0.0, t_max, n_steps,endpoint=True, retstep=True)


L2 = 4  # system size
N_up2 = L // 2 + L % 2  # number of fermions with spin up
N_down2 = L // 2  # number of fermions with spin down
# N_down = 0
N2 = N_up + N_down  # number of particles
t02 = 0.52  # hopping strength
# U = 0*t0  # interaction strength
U2 = 1 * t0  # interaction strength
gamma2 = 0.01
mu2 = 0.1
pbc2 = False

t_max2 = 50
n_steps2 = 500
times2, delta2 = np.linspace(0.0, t_max, n_steps, endpoint=True, retstep=True)
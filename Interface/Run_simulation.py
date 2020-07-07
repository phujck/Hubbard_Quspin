##########################################################
# Hubbard model simulation with interface and            #
#  spin- orbit coupling. For details on SO COUPLING, see #
# Z. Phys. B - Condensed Matter 49, 313-317 (1983)       #
##########################################################
from __future__ import print_function, division
import os
import sys

"""Open MP and MKL should speed up the time required to run these simulations!"""
# threads = sys.argv[1]
threads = 6
os.environ['OMP_NUM_THREADS'] = '{}'.format(threads)
os.environ['MKL_NUM_THREADS'] = '{}'.format(threads)
# line 4 and line 5 below are for development purposes and can be removed

from quspin.operators import hamiltonian, exp_op, quantum_operator  # operators
from quspin.basis import spinful_fermion_basis_1d  # Hilbert space basis
from quspin.tools.measurements import obs_vs_time  # calculating dynamics
import numpy as np  # general math functions
from scipy.sparse.linalg import eigsh
from time import time  # tool for calculating computation time
import matplotlib.pyplot as plt  # plotting library

sys.path.append('../')
from tools import parameter_instantiate as hhg  # Used for scaling units.

t_init = time()

"""Hubbard model Parameters"""
L = 8  # system size
N_up = L // 2  # number of fermions with spin up
N_down = L // 2  # number of fermions with spin down
# N_up = 5  # number of fermions with spin up
# N_down = 5  # number of fermions with spin down
N = N_up + N_down  # number of particles
t0 = 0.52  # hopping strength
SO = 0 * t0  # spin orbit coupling
# U = 0*t0  # interaction strength
U = 1 * t0  # interaction strength
pbc = True

"""Set up the partition of the system's onsite potentials"""
U_a = 0 * t0
U_b = 0.00 * t0
partition = 5
U = []
for n in range(L):
    # if n < int(nx/2):
    if n < partition:
        U.append(U_a)
    else:
        U.append(U_b)
U = np.array(U)

"""Laser pulse parameters"""
field = 32.9  # field angular frequency THz
F0 = 10  # Field amplitude MV/cm
a = 4  # Lattice constant Angstroms

"""instantiate parameters with proper unit scaling. In this case everything is scaled to units of t_0"""
lat = hhg(field=field, nup=N_up, ndown=N_down, nx=L, ny=0, U=U, SO=SO, t=t0, F0=F0, a=a, pbc=pbc)

"""Define e^i*phi for later dynamics. Important point here is that for later implementations of tracking, we
will pass phi as a global variable that will be modified as appropriate during evolution"""


def phi(current_time):
    phi = (lat.a * lat.F0 / lat.field) * (np.sin(lat.field * current_time / (2. * cycles)) ** 2.) * np.sin(
        lat.field * current_time)
    return phi


def expiphi(current_time):
    phi = (lat.a * lat.F0 / lat.field) * (np.sin(lat.field * current_time / (2. * cycles)) ** 2.) * np.sin(
        lat.field * current_time)
    return np.exp(1j * phi)


def expiphiconj(current_time):
    phi = (lat.a * lat.F0 / lat.field) * (np.sin(lat.field * current_time / (2. * cycles)) ** 2.) * np.sin(
        lat.field * current_time)
    return np.exp(-1j * phi)


"""This is used for setting up Hamiltonian in Quspin."""
dynamic_args = []

"""System Evolution Time"""
cycles = 10  # time in cycles of field frequency
n_steps = 2000
start = 0
stop = cycles / lat.freq
# stop = 0.5
times, delta = np.linspace(start, stop, num=n_steps, endpoint=True, retstep=True)
# print(times)

"""set up parameters for saving expectations later"""
outfile = './Data/expectations:{}sites-{}up-{}down-{}t0-{}U-{}SO-{}cycles-{}steps-{}pbc.npz'.format(L, N_up, N_down, t0,
                                                                                                    U, SO, cycles,
                                                                                                    n_steps, pbc)

"""create basis"""
# build spinful fermions basis. It's possible to specify certain symmetry sectors here, but I'm not going to touch that
# until I understand it better.
basis = spinful_fermion_basis_1d(L, Nf=(N_up, N_down))
# basis = spinful_fermion_basis_1d(L, Nf=(N_up, N_down),sblock=1,kblock=1)

#
"""building model"""
###########################################################################
# to give you a very rough idea, the Hamiltonian breaks down as           #
# H=(e^(-i*phi)(-t0 + i*sigma*SO)*left_hopping ops + H.C) + U*n_up*n_down #
###########################################################################
# define site-coupling lists
# if max(lat.U):
#     int_list = [[lat.U[i], i, i] for i in range(L)]  # onsite interaction
if max(lat.U):
    int_list = [[lat.U[i], i, i] for i in range(L)]  # onsite interaction

    # create static lists
    # Note that the pipe determines the spinfulness of the operator. | on the left corresponds to down spin, | on the right
    # is for up spin. For the onsite interaction here, we have:
    static_Hamiltonian_list = [
        ["n|n", int_list],  # onsite interaction
    ]
else:
    static_Hamiltonian_list = []

"""add dynamic lists for hopping"""
if SO:
    print('spin orbit coupling active')
    up_param = lat.t + 1j * lat.SO
    down_param = up_param.conjugate()

    hop_right_up = [[up_param, i, i + 1] for i in range(L - 1)]  # hopping to the right, up spin OBC
    hop_right_down = [[down_param, i, i + 1] for i in range(L - 1)]  # hopping to the right, down spin OBC
    hop_left_up = [[-down_param, i, i + 1] for i in range(L - 1)]  # hopping to the left, up spin OBC
    hop_left_down = [[-up_param, i, i + 1] for i in range(L - 1)]  # hopping to the left, up spin OBC
    hop_right = [[lat.t, i, i + 1] for i in range(L - 1)]  # hopping to the right, down spin OBC
    hop_left = [[-lat.t, i, i + 1] for i in range(L - 1)]  # hopping to the left, up spin OBC
    """Add periodic boundaries"""
    if lat.pbc:
        hop_right_up.append([up_param, L - 1, 0])
        hop_right_down.append([down_param, L - 1, 0])

        hop_left_up.append([-down_param, L - 1, 0])
        hop_left_down.append([-up_param, L - 1, 0])

        hop_right.append([lat.t, L - 1, 0])
        hop_left.append([-lat.t, L - 1, 0])

        dynamic_Hamiltonian_list = [
            ["+-|", hop_left_up, expiphiconj, dynamic_args],  # up hop left
            ["-+|", hop_right_up, expiphi, dynamic_args],  # up hop right
            ["|+-", hop_left_down, expiphiconj, dynamic_args],  # down hop left
            ["|-+", hop_right_down, expiphi, dynamic_args],  # down hop right
        ]
else:
    hop_right = [[lat.t, i, i + 1] for i in range(L - 1)]  # hopping to the right, down spin OBC
    hop_left = [[-lat.t, i, i + 1] for i in range(L - 1)]  # hopping to the left, up spin OBC
    """Add periodic boundaries"""
    if lat.pbc:
        hop_right.append([lat.t, L - 1, 0])
        hop_left.append([-lat.t, L - 1, 0])
        dynamic_Hamiltonian_list = [
            ["+-|", hop_left, expiphiconj, dynamic_args],  # up hop left
            ["-+|", hop_right, expiphi, dynamic_args],  # up hop right
            ["|+-", hop_left, expiphiconj, dynamic_args],  # down hop left
            ["|-+", hop_right, expiphi, dynamic_args],  # down hop right
        ]

# After creating the site lists, we attach an operator and a time-dependent function to them


"""build the Hamiltonian for actually evolving this bastard."""
# Hamiltonian builds an operator, the first argument is always the static operators, then the dynamic operators.
ham = hamiltonian(static_Hamiltonian_list, dynamic_Hamiltonian_list, basis=basis)

"""build up the other operator expectations here as a dictionary"""
operator_dict = dict(H=ham)
no_checks = dict(check_pcon=False, check_symm=False, check_herm=False)
# hopping operators for building current. Note that the easiest way to build an operator is just to cast it as an
# instance of the Hamiltonian class. Note in this instance the hops up and down have the e^iphi factor attached directly
operator_dict["lhopup"] = hamiltonian([], [["+-|", hop_left, expiphiconj, dynamic_args]], basis=basis, **no_checks)
operator_dict["lhopdown"] = hamiltonian([], [["|+-", hop_left, expiphiconj, dynamic_args]], basis=basis, **no_checks)
# Add individual spin expectations
for j in range(L):
    # spin up densities for each site
    operator_dict["nup" + str(j)] = hamiltonian([["n|", [[1.0, j]]]], [], basis=basis, **no_checks)
    # spin down
    operator_dict["ndown" + str(j)] = hamiltonian([["|n", [[1.0, j]]]], [], basis=basis, **no_checks)
    # doublon densities
    operator_dict["D" + str(j)] = hamiltonian([["n|n", [[1.0, j, j]]]], [], basis=basis, **no_checks)

"""build ground state"""
# TODO IMPLEMENTING AN IMAGINARY TIME EVOLUTION FOR THE GROUND STATE.
########################################################################################################################
# GIANT HEALTH WARNING. FOR SOME SITE NUMBERS AND FILLINGS, THIS GROUND STATE IS HIGHLY DEGENERATE. THIS METHOD ISN'T  #
# GUARANTEED TO GIVE YOU A GROUND STATE SYMMETRIC IN THE SPINS! THIS IS A PARTICULAR ISSUE FOR L=8 AND HALF FILLING!   #
# WORKAROUNDS ARE TO INCLUDE A TINY AMOUNT OF U OR SPIN ORBIT.                                                         #
########################################################################################################################

print("calculating ground state")
E, psi_0 = ham.eigsh(k=1, which='SA')

#################################################
# code that messed around exploring groundstate #
#################################################
# E, psi_0 = ham.eigsh(which='SA')
# print(E)
# print(psi_0.shape)
# print(E)
# psi_0=psi_0[:,1]
# psi_0=np.sum(psi_0, axis=1)/2

# apparently you can get a speedup for the groundstate calculation using this method with multithread. Note that it's
# really not worth it unless you your number of sites gets _big_, and even then it's the time evolution which is going
# to kill you:
# E, psi_0 = eigsh(ham.aslinearoperator(time=0), k=1, which='SA')

print("ground state calculated, energy is {:.2f}".format(E[0]))
# psi_0.reshape((-1,))
# psi_0=psi_0.flatten
print('evolving system')
ti = time()
"""evolving system. In this simple case we'll just use the built in solver"""
# this version returns the generator for psi
# psi_t=ham.evolve(psi_0,0.0,times,iterate=True)

# this version returns psi directly, last dimension contains time dynamics. The squeeze is necessary for the
# obs_vs_time to work properly
psi_t = ham.evolve(psi_0, 0.0, times)
psi_t = np.squeeze(psi_t)
print("Evolution done! This one took {:.2f} seconds".format(time() - ti))

# calculate the expectations for every bastard in the operator dictionary
ti = time()
# note that here the expectations
expectations = obs_vs_time(psi_t, times, operator_dict)
print(type(expectations))
current_partial = (expectations['lhopup'] + expectations['lhopdown'])
current = -1j * lat.a * (current_partial - current_partial.conjugate())
expectations['current'] = current
print("Expectations calculated! This took {:.2f} seconds".format(time() - ti))

print("Saving Expectations. We have {} of them".format(len(expectations)))
print('Expectations are')
print(expectations.keys())
np.savez(outfile, **expectations)
print('All finished. Total time was {:.2f} seconds using {:d} threads'.format((time() - t_init), threads))

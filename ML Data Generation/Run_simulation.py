##########################################################
# Very basic Hubbard model simulation. Should be used as #
# a base to build on with e.g. interfaces, S.O. coupling #
# tracking and low-rank approximations. Should be fairly #
# self-explanatory, and is based on the Quspin package.  #
# See:  http://weinbe58.github.io/QuSpin/index.html      #
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
import itertools

t_init = time()

"""Hubbard model Parameters"""
L = 6  # system size
N_up = L // 2 + L % 2  # number of fermions with spin up
N_down = L // 2  # number of fermions with spin down
N = N_up + N_down  # number of particles
t0 = 0.52  # hopping strength
# U = 0*t0  # interaction strength
U = 0 * t0  # interaction strength
pbc = True

"""Laser pulse parameters"""
field = 32.9  # field angular frequency THz
F0 = 10  # Field amplitude MV/cm
a = 4  # Lattice constant Angstroms

"""instantiate parameters with proper unit scaling"""
lat = hhg(field=field, nup=N_up, ndown=N_down, nx=L, ny=0, U=U, t=t0, F0=F0, a=a, pbc=pbc)

# """Make a parameter dictionary"""
# params = dict()
# params['field'] = lat.field
# params['sites'] = lat.nx
# params['U'] = lat.U
# params['t0'] = lat.t
# params['F0'] = lat.F0
# params['a'] = lat.a
# params['pbc'] = lat.pbc
# print(params)

"""This is used for setting up Hamiltonian in Quspin."""
dynamic_args = []

"""System Evolution Time"""
cycles = 10  # time in cycles of field frequency
n_steps = 2000
start = 0
stop = cycles / lat.freq
times, delta = np.linspace(start, stop, num=n_steps, endpoint=True, retstep=True)

"""set up parameters for saving expectations later"""

"""Define e^i*phi for later dynamics. We use global variables for the parameters we're going to vary. This is essential
to save us from needing to rebuild the Hamiltonian for each iteration."""

"""need to have these lists for building the initial ground-state, but their precise form is irrelevant"""
prefactor_list_sin = np.zeros(1)
prefactor_list_cos = np.zeros(1)
frequency_list_sin = np.zeros(1)
frequency_list_cos = np.zeros(1)


def phi(current_time):
    phi = (lat.a * lat.F0 / lat.field) * (np.sin(lat.field * current_time / (2. * cycles)) ** 2.) * np.sum(
        prefactor_list_cos * np.cos(frequency_list_cos * lat.field * current_time) + prefactor_list_sin * np.sin(
            frequency_list_sin * lat.field * current_time)
    )
    return phi


def expiphi(current_time):
    return np.exp(1j * phi(current_time))


def expiphiconj(current_time):
    return np.exp(-1j * phi(current_time))


"""create basis"""
# build spinful fermions basis. It's possible to specify certain symmetry sectors here, but I'm not going to touch that
# until I understand it better.
basis = spinful_fermion_basis_1d(L, Nf=(N_up, N_down))
#
"""building model"""
# define site-coupling lists
int_list = [[lat.U, i, i] for i in range(L)]  # onsite interaction

# create static lists
# Note that the pipe determines the spinfulness of the operator. | on the left corresponds to down spin, | on the right
# is for up spin. For the onsite interaction here, we have:
static_Hamiltonian_list = [
    ["n|n", int_list],  # onsite interaction
]

# add dynamic lists
hop_right = [[lat.t, i, i + 1] for i in range(L - 1)]  # hopping to the right OBC
hop_left = [[-lat.t, i, i + 1] for i in range(L - 1)]  # hopping to the left OBC

"""Add periodic boundaries"""
if lat.pbc:
    hop_right.append([lat.t, L - 1, 0])
    hop_left.append([-lat.t, L - 1, 0])

# After creating the site lists, we attach an operator and a time-dependent function to them
dynamic_Hamiltonian_list = [
    ["+-|", hop_left, expiphiconj, dynamic_args],  # up hop left
    ["-+|", hop_right, expiphi, dynamic_args],  # up hop right
    ["|+-", hop_left, expiphiconj, dynamic_args],  # down hop left
    ["|-+", hop_right, expiphi, dynamic_args],  # down hop right
]

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
# for j in range(L):
#     # spin up densities for each site
#     operator_dict["nup" + str(j)] = hamiltonian([["n|", [[1.0, j]]]], [], basis=basis, **no_checks)
#     # spin down
#     operator_dict["ndown" + str(j)] = hamiltonian([["|n", [[1.0, j]]]], [], basis=basis, **no_checks)
#     # doublon densities
#     operator_dict["D" + str(j)] = hamiltonian([["n|n", [[1.0, j, j]]]], [], basis=basis, **no_checks)

"""set parameters for generating data"""
# open question for generating data sets- is it better to span some parameter space, of just randomly sample it to
# generate the set? Random generation is easier to implement here, but I'm worried it may induce a bias in any model
# learned from that data set.

# part of the problem with HHG is that it leads to very spiky output data. Use this to tamp it down
harmonic_desired = 100

degree = 5  # set the degree of the sinusoidal sum
max_prefactor = 10  # Set the maximum value each prefactor can be.
# This is here to ensure we don't generate a frequency higher than the Nyquist freq.
max_freq = np.pi / (delta * lat.freq * harmonic_desired)
# print('maximumfrequency')
# print(max_freq)
# print('maximum frequency')
print('maximum frequency sampled is {:.2f} omega_0'.format(max_freq))
print('maximum field amplitude used is pm {:.2f} F_0'.format(max_prefactor))
realisations = 1024

phi_list = []
current_list = []
theta_list = []
R_list = []
E_list = []
sin_prefactors = []
cos_prefactors = []
sin_frequencies = []
cos_frequencies = []

"""build ground state"""
print("calculating ground state")
E, psi_0 = ham.eigsh(k=1, which='SA')
# apparently you can get a speedup for the groundstate calculation using this method with multithread. Note that it's
# really not worth it unless you your number of sites gets _big_, and even then it's the time evolution which is going
# to kill you:
# E, psi_0 = eigsh(ham.aslinearoperator(time=0), k=1, which='SA')

print("ground state calculated, energy is {:.2f}".format(E[0]))
# psi_0.reshape((-1,))
# psi_0=psi_0.flatten


"""run the evolution for different realisations of the parameters"""
for n in range(realisations):
    t_init_realisation = time()
    """seed the parameters and add them to their appropriate lists"""
    np.random.seed()
    prefactor_list_sin = np.random.uniform(-max_prefactor, max_prefactor, degree)
    sin_prefactors.append(prefactor_list_sin)
    prefactor_list_cos = np.random.uniform(-max_prefactor, max_prefactor, degree)
    cos_prefactors.append(prefactor_list_sin)
    frequency_list_sin = np.random.uniform(0, max_freq, degree)
    sin_frequencies.append(frequency_list_sin)
    frequency_list_cos = np.random.uniform(0, max_freq, degree)
    cos_frequencies.append(frequency_list_cos)
    psi_new = psi_0.copy()
    """generate phi and add it to its appropriate list"""
    phi_as_list = [phi(time) for time in times]
    phi_list.append(phi_as_list)
    # this version returns psi directly, last dimension contains time dynamics. The squeeze is necessary for the
    # obs_vs_time to work properly
    print('evolving system')
    """evolving system. In this simple case we'll just use the built in solver"""
    # this version returns the generator for psi
    # psi_t=ham.evolve(psi_0,0.0,times,iterate=True)
    ti = time()
    psi_t = ham.evolve(psi_new, 0.0, times)
    psi_t = np.squeeze(psi_t)
    print("Evolution done! This one took {:.2f} seconds".format(time() - ti))
    # calculate the expectations for every bastard in the operator dictionary
    ti = time()
    # note that here the expectations
    expectations = obs_vs_time(psi_t, times, operator_dict)
    current_partial = (expectations['lhopup'] + expectations['lhopdown'])
    theta = np.angle(current_partial)
    theta_list.append(theta)
    R = np.abs(current_partial)
    R_list.append(R)
    current = -1j * lat.a * (current_partial - current_partial.conjugate())
    current_list.append(current)
    E_list.append(expectations['H'])
    print("Expectations calculated! This took {:.2f} seconds".format(time() - ti))
    print('Realisation {}/{} Done. Total time was {:.2f} seconds using {:d} threads'.format(n + 1, realisations, (
            time() - t_init_realisation), threads))

outstr = ':{}realisations-{}sites-{}up-{}down-{}t0-{}U-{}cycles-{}steps-{}maxfrequency-{}maxprefactor-{}pbc.npz'.format(
    realisations, L, N_up, N_down, t0, U, cycles, n_steps, max_freq, max_prefactor, pbc)
# print("Saving Expectations. We have {} of them".format(len(expectations)))
# np.savez(outfile, **expectations)

print('All finished. Total time was {:.2f} seconds using {:d} threads'.format((time() - t_init), threads))

"""stack the data up. First dimension is time, second is realisation"""
phi_list = np.vstack(phi_list).T
current_list = np.vstack(current_list).T
theta_list = np.vstack(theta_list).T
R_list = np.vstack(R_list).T
E_list = np.vstack(E_list).T
sin_prefactors = np.vstack(sin_prefactors).T
cos_prefactors = np.vstack(cos_prefactors).T
sin_frequencies = np.vstack(sin_frequencies).T
cos_frequencies = np.vstack(cos_frequencies).T

"""Finally, let's save the data"""
np.savez('./Data/phi' + outstr, phi_list)
np.savez('./Data/current' + outstr, current_list)
np.savez('./Data/theta' + outstr, theta_list)
np.savez('./Data/R' + outstr, R_list)
np.savez('./Data/energies' + outstr, E_list)
np.savez('./Data/sin_prefactors' + outstr, sin_prefactors)
np.savez('./Data/cos_prefactors' + outstr, cos_prefactors)
np.savez('./Data/sin_frequencies' + outstr, sin_frequencies)
np.savez('./Data/cos_frequencies' + outstr, cos_frequencies)
print(phi_list.shape)

# """quick check to see we've been generating different phi"""
# plt.plot(times, phi_list)
# plt.show()
#
# plt.plot(times,current_list)
# plt.show()

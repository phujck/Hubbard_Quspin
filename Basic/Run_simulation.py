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
threads = 16
os.environ['NUMEXPR_MAX_THREADS']='{}'.format(threads)
os.environ['NUMEXPR_NUM_THREADS']='{}'.format(threads)
os.environ['OMP_NUM_THREADS'] = '{}'.format(threads)
os.environ['MKL_NUM_THREADS'] = '{}'.format(threads)
# line 4 and line 5 below are for development purposes and can be remove
from quspin.operators import hamiltonian, exp_op, quantum_operator  # operators
from quspin.basis import spinful_fermion_basis_1d  # Hilbert space basis
from quspin.tools.measurements import obs_vs_time  # calculating dynamics
from quspin.tools.evolution import evolve  #evolving system
import numpy as np  # general math functions
from scipy.sparse.linalg import eigsh
from time import time  # tool for calculating computation time
from tqdm import tqdm
import matplotlib.pyplot as plt  # plotting library
from quspin.tools.measurements import project_op

sys.path.append('../')
from tools import parameter_instantiate as hhg  # Used for scaling units.
import psutil
# # note cpu_count for logical=False returns the wrong number for multi-socket CPUs.
print("logical cores available {}".format(psutil.cpu_count(logical=True)))
t_init = time()
np.__config__.show()
"""Hubbard model Parameters"""
L = 8# system size
N_up = L // 2 + L % 2  # number of fermions with spin up
N_down = L // 2  # number of fermions with spin down
N = N_up + N_down  # number of particles
t0 = 0.52  # hopping strength
# U = 0*t0  # interaction strength
U = 0.3* t0  # interaction strength
pbc = True

"""Laser pulse parameters"""
field = 32.9  # field angular frequency THz
F0 = 10  # Field amplitude MV/cm
a = 4  # Lattice constant Angstroms

"""instantiate parameters with proper unit scaling"""
lat = hhg(field=field, nup=N_up, ndown=N_down, nx=L, ny=0, U=U, t=t0, F0=F0, a=a, pbc=pbc)
"""Define e^i*phi for later dynamics. Important point here is that for later implementations of tracking, we
will pass phi as a global variable that will be modified as appropriate during evolution"""

def phi(current_time):
    phi = (lat.a * lat.F0 / lat.field) * (np.sin(lat.field * current_time / (2. * cycles)) ** 2.) * np.sin(
        lat.field * current_time)
    return phi
def expiphi(current_time):

    return np.exp(1j * phi(current_time))


def expiphiconj(current_time):

    return np.exp(-1j * phi(current_time))


"""This is used for setting up Hamiltonian in Quspin."""
dynamic_args = []

"""System Evolution Time"""
cycles = 10  # time in cycles of field frequency
n_steps = 2000
start = 0
stop = cycles / lat.freq
times, delta = np.linspace(start, stop, num=n_steps, endpoint=True, retstep=True)

"""set up parameters for saving expectations later"""
outfile = './Data/expectations:{}sites-{}up-{}down-{}t0-{}U-{}cycles-{}steps-{}pbc.npz'.format(L, N_up, N_down,
                                                                                                     t0, U, cycles,
                                                                                                     n_steps, pbc)

"""create basis"""
# build spinful fermions basis. It's possible to specify certain symmetry sectors here, but I'm not going to touch that
# until I understand it better.
basis = spinful_fermion_basis_1d(L, Nf=(N_up, N_down))
# basis = spinful_fermion_basis_1d(L, Nf=(N_up, N_down),a=1,sblock=1,kblock=1)
# basis = spinful_fermion_basis_1d(L, Nf=(N_up, N_down),sblock=1)

print('Hilbert space size: {0:d}.\n'.format(basis.Ns))

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
operator_dict['neighbour']= hamiltonian([["+-|", hop_left],["|+-", hop_left]],[], basis=basis, **no_checks)
operator_dict["lhopup"] = hamiltonian([], [["+-|", hop_left, expiphiconj, dynamic_args]], basis=basis, **no_checks)
operator_dict["lhopdown"] = hamiltonian([], [["|+-", hop_left, expiphiconj, dynamic_args]], basis=basis, **no_checks)
# #Add individual spin expectations
for j in range(L):
    # spin up densities for each site
    operator_dict["nup" + str(j)] = hamiltonian([["n|", [[1.0, j]]]], [], basis=basis, **no_checks)
    # spin down
    operator_dict["ndown" + str(j)] = hamiltonian([["|n", [[1.0, j]]]], [], basis=basis, **no_checks)
    # doublon densities
    operator_dict["D" + str(j)] = hamiltonian([["n|n", [[1.0, j, j]]]], [], basis=basis, **no_checks)

"""build ground state"""
print("calculating ground state")
E, psi_0 = ham.eigsh(k=1, which='SA')
# E, psi_0 = ham.eigh(time=0)

# apparently you can get a speedup for the groundstate calculation using this method with multithread. Note that it's
# really not worth it unless you your number of sites gets _big_, and even then it's the time evolution which is going
# to kill you:
# E, psi_0 = eigh(ham.aslinearoperator(time=0), k=1, which='SA')


# alternate way of doing this
# # psi_0=np.ones(ham.Ns)
# psi_0=np.random.random(ham.Ns)
# def imag_time(tau,phi):
#
# 	return -( ham.dot(phi,time=0))
# taus=np.linspace(0,100,100)
# psi_imag = evolve(psi_0, taus[0], taus, imag_time, iterate=False, atol=1E-12, rtol=1E-12,verbose=True,imag_time=True)
# print(psi_imag.shape)
# psi_0=psi_imag[:,-1]
print(E)
print(psi_0.shape)
print('normalisation')
# psi_0=psi_0[:,2]
print(np.linalg.norm(psi_0))
psi_0=psi_0/np.linalg.norm(psi_0)
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
psi_t = ham.evolve(psi_0, 0.0, times,verbose=True)
psi_t = np.squeeze(psi_t)
print("Evolution done! This one took {:.2f} seconds".format(time() - ti))
# calculate the expectations for every bastard in the operator dictionary
ti = time()
# note that here the expectations
expectations = obs_vs_time(psi_t, times, operator_dict)
print(type(expectations))
current_partial = (expectations['lhopup'] + expectations['lhopdown'])
current = 1j * lat.a * (current_partial - current_partial.conjugate())
expectations['current'] = current
expectations['phi']=phi(times)
print("Expectations calculated! This took {:.2f} seconds".format(time() - ti))

print("Saving Expectations. We have {} of them".format(len(expectations)))
np.savez(outfile, **expectations)

print('All finished. Total time was {:.2f} seconds using {:d} threads'.format((time() - t_init), threads))
# npzfile = np.load(outfile)
# print('npzfile.files: {}'.format(npzfile.files))
# print('npzfile["1"]: {}'.format(npzfile["current"]))
# newh=npzfile['H']
# doublon=np.zeros(len(times))
#
# times=times*lat.freq
# plt.plot(times,newh)
# plt.show()

# plt.plot(times,current)
# # plt.plot(times,current_partial)
# plt.show()
# for j in range(L):
#     plt.plot(times, expectations["ndown"+str(j)])
# plt.show()

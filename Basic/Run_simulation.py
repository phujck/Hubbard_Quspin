from __future__ import print_function, division
import os

"""Open MP and MKL should speed up the time required to run these simulations!"""
threads = 5
os.environ['OMP_NUM_THREADS'] = '{}'.format(threads)
os.environ['MKL_NUM_THREADS'] = '{}'.format(threads)
# line 4 and line 5 below are for development purposes and can be removed

from quspin.operators import hamiltonian, exp_op, quantum_operator  # operators
from quspin.basis import spinful_fermion_basis_1d  # Hilbert space basis
from quspin.tools.measurements import obs_vs_time  # calculating dynamics
import numpy as np  # general math functions
from time import time  # tool for calculating computation time
import matplotlib.pyplot as plt  # plotting library


class hhg:
    def __init__(self, field, nup, ndown, nx, ny, U, t=0.52, F0=10., a=4., lat_type='square', pbc=True):
        self.nx = nx
        self.pbc = pbc
        if pbc:
            print("Periodic Boundary conditions")
        else:
            print("Open Boundary conditions")
        self.nup = nup
        print("%s up electrons" % self.nup)
        self.ndown = ndown
        print("%s down electrons" % self.nup)
        self.ne = nup + ndown
        # input units: THz (field), eV (t, U), MV/cm (peak amplitude), Angstroms (lattice cst)
        # converts to a'.u, which are atomic units but with energy normalised to t, so
        # Note, hbar=e=m_e=1/4pi*ep_0=1, and c=1/alpha=137
        print("Scaling units to energy of t_0")
        factor = 1. / (t * 0.036749323)
        # factor=1
        self.factor = factor
        # self.factor=1
        self.U = U / t
        print("U= %.3f t_0" % self.U)
        # self.U=U
        self.t = 1.
        print("t_0 = %.3f" % self.t)
        # self.t=t
        # field is the angular frequency, and freq the frequency = field/2pi
        self.field = field * factor * 0.0001519828442
        print("angular frequency= %.3f" % self.field)
        self.freq = self.field / (2. * 3.14159265359)
        print("frequency= %.3f" % self.freq)
        self.a = (a * 1.889726125) / factor
        print("lattice constant= %.3f" % self.a)
        self.F0 = F0 * 1.944689151e-4 * (factor ** 2)
        print("Field Amplitude= %.3f" % self.F0)
        assert self.nup <= self.nx, 'Too many ups!'
        assert self.ndown <= self.nx, 'Too many downs!'


"""Hubbard model Parameters"""
L = 10  # system size
N_up = L // 2 + L % 2  # number of fermions with spin up
N_down = L // 2  # number of fermions with spin down
N = N_up + N_down  # number of particles
t0 = 0.52  # hopping strength
# U = 0*t0  # interaction strength
U = 1 * t0  # interaction strength
pbc = False

"""Laser pulse parameters"""
field = 32.9  # field angular frequency THz
F0 = 10  # Field amplitude MV/cm
a = 4  # Lattice constant Angstroms
lat = hhg(field=field, nup=N_up, ndown=N_down, nx=L, ny=0, U=U, t=t0, F0=F0, a=a, pbc=pbc)

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


dynamic_args = []

"""System Evolution Time"""
cycles = 10  # time in cycles of field frequency
n_steps = 1000
start = 0
stop = cycles / lat.freq
times, delta = np.linspace(start, stop, num=n_steps, endpoint=True, retstep=True)


"""set up parameters for saving expectations later"""
outfile = './Basic/expectations:{}sites-{}up-{}down-{}t0-{}U-{}cycles-{}steps.npz'.format(L, N_up, N_down, t0, U, cycles,
                                                                                  n_steps)

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
# is for up spin. For the onsite interaction here
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

# After creating the site lists, we attach an operator and a time-dependent function to them!
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
for j in range(L):
    operator_dict["nup" + str(j)] = hamiltonian([["n|", [[1.0, j]]]], [], basis=basis, **no_checks)
    operator_dict["ndown" + str(j)] = hamiltonian([["|n", [[1.0, j]]]], [], basis=basis, **no_checks)
    operator_dict["D" + str(j)] = hamiltonian([["n|n", [[1.0, j, j]]]], [], basis=basis, **no_checks)

"""build ground state"""
print("calculating ground state")
# apparently MKL multithreading improves the speedup here
E, psi_0 = ham.eigsh(k=1, which='SA')
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

print("saving expectations")
np.savez(outfile, **expectations)

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

##########################################################
# Implementing tracking control.                         #
# For Further details see Phys. Rev. Lett. 124, 183201   #
##########################################################
from __future__ import print_function, division
import os
import sys

"""Open MP and MKL should speed up the time required to run these simulations!"""
# threads = sys.argv[1]
threads = 6
os.environ['OMP_NUM_THREADS'] = '{}'.format(threads)
os.environ['MKL_NUM_THREADS'] = '{}'.format(threads)
# line 4 and line 5 below are for development purposes and can be remove
from quspin.operators import hamiltonian, exp_op, quantum_operator  # operators
from quspin.basis import spinful_fermion_basis_1d  # Hilbert space basis
from quspin.tools.measurements import obs_vs_time  # calculating dynamics
from quspin.tools.evolution import evolve # ODE evolve tool
import numpy as np  # general math functions
from scipy.sparse.linalg import eigsh
from time import time  # tool for calculating computation time
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt  # plotting library
from tqdm import tqdm
sys.path.append('../')
from tools import parameter_instantiate as hhg  # Used for scaling units.
import psutil
# note cpu_count for logical=False returns the wrong number for multi-socket CPUs.
print("logical cores available {}".format(psutil.cpu_count(logical=True)))
t_init = time()
np.__config__.show()
""" Original Hubbard model Parameters"""
L = 6# system size
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
a = 4  # Lattice constant Angstrom

""" Tracking Hubbard model Parameters"""
a_scale=1 #scaling lattic parameter for tracking
J_scale=1  #scaling J for tracking.
L_track = 6# system size
N_up_track = L_track // 2 + L % 2  # number of fermions with spin up
N_down_track = L_track // 2  # number of fermions with spin down
N = N_up_track + N_down_track  # number of particles
t0_track = 0.52  # hopping strength
# U = 0*t0  # interaction strength
U_track = 0 * t0  # interaction strength
pbc_track = True
a_track=a*a_scale

"""instantiate parameters with proper unit scaling"""
lat = hhg(field=field, nup=N_up, ndown=N_down, nx=L, ny=0, U=U, t=t0, F0=F0, a=a, pbc=pbc)
lat_track = hhg(field=field, nup=N_up_track, ndown=N_down_track, nx=L_track, ny=0, U=U_track, t=t0_track, F0=F0, a=a_track, pbc=pbc_track)

"""This is used for setting up Hamiltonian in Quspin."""
dynamic_args = []

"""System Evolution Time"""
cycles = 10  # time in cycles of field frequency
n_steps = 2000
start = 0
stop = cycles / lat.freq
times, delta = np.linspace(start, stop, num=n_steps, endpoint=True, retstep=True)


"""load original data"""
loadfile = '../Basic/Data/expectations:{}sites-{}up-{}down-{}t0-{}U-{}cycles-{}steps-{}pbc.npz'.format(L, N_up, N_down,
                                                                                                     t0, U, cycles,
                                                                                                     n_steps, pbc)
expectations = dict(np.load(loadfile))
J_field=expectations["current"]
phi_original=expectations["phi"]
"""Interpolates the current to be tracked."""
J_target = interp1d(times, J_scale*J_field, fill_value=0, bounds_error=False, kind='cubic')


"""create basis"""
# build spinful fermions basis. It's possible to specify certain symmetry sectors here, but I'm not going to touch that
# until I understand it better.
basis = spinful_fermion_basis_1d(L, Nf=(N_up, N_down))
#
"""building model"""
# define site-coupling lists
int_list = [[lat_track.U, i, i] for i in range(L)]  # onsite interaction

# create static lists
# Note that the pipe determines the spinfulness of the operator. | on the left corresponds to down spin, | on the right
# is for up spin. For the onsite interaction here, we have:
static_Hamiltonian_list = [
    ["n|n", int_list],  # onsite interaction
]

# hopping operator lists.
hop_right = [[-1, i, i + 1] for i in range(L - 1)]  # hopping to the right OBC
hop_left = [[1, i, i + 1] for i in range(L - 1)]  # hopping to the left OBC

"""Add periodic boundaries"""
if lat.pbc:
    hop_right.append([-1, L - 1, 0])
    hop_left.append([1, L - 1, 0])

# After creating the site lists, we attach an operator and a time-dependent function to them
no_checks = dict(check_pcon=False, check_symm=False, check_herm=False)

hop_left_op= hamiltonian([["+-|", hop_left],["|+-", hop_left]],[], basis=basis, **no_checks)  # left hoperators
hop_right_op=hop_left_op.getH() #right hoperators
# hop_right_op= hamiltonian([["-+|", hop_right],["|-+", hop_right]],[], basis=basis, **no_checks)  # alternate r hoperator


"""build the Hamiltonian for actually evolving this bastard."""
# Hamiltonian builds an operator, the first argument is always the static operators, then the dynamic operators.
ham_onsite = hamiltonian(static_Hamiltonian_list, [], basis=basis) #here we just use the onsite Hamiltonian
ham_init=ham_onsite -lat_track.t*(hop_left_op+hop_right_op)
operator_dict = dict(H_onsite=ham_onsite)




"""build ground state"""
print("calculating ground state")
E, psi_0 = ham_init.eigsh(k=1, which='SA')
psi_0 = np.squeeze(psi_0)

# apparently you can get a speedup for the groundstate calculation using this method with multithread. Note that it's
# really not worth it unless you your number of sites gets _big_, and even then it's the time evolution which is going
# to kill you:
# E, psi_0 = eigsh(ham.aslinearoperator(time=0), k=1, which='SA')

print("ground state calculated, energy is {:.2f}".format(E[0]))
# psi_0.reshape((-1,))
# psi_0=psi_0.flatten
print('evolving system')
ti = time()

"""set up what we need for tracking."""

def phi_J_track(lat, current_time, J_target, nn_expec):
    # Import the current function
    # if current_time <self.delta:
    #     current=self.J_reconstruct(0)
    # else:
    #     current = self.J_reconstruct(current_time-self.delta)
    current = J_target(current_time)
    # Arrange psi to calculate the nearest neighbour expectations
    D = nn_expec
    # D=hop_left_op.expt_value(psi_t) #alternative, calculating expectation inside
    # print(D)
    angle = np.angle(D)
    # print(angle)
    mag = np.abs(D)
    # print(mag)
    scalefactor = 2 * lat.a * lat.t * mag
    # assert np.abs(current)/scalefactor <=1, ('Current too large to reproduce, ration is %s' % np.abs(current/scalefactor))
    arg = -current / (2 * lat.a * lat.t * mag)
    phi = np.arcsin(arg + 0j) + angle
    # phi = np.arcsin(arg + 0j)
    # Solver is sensitive to whether we specify phi as real or not!
    phi=phi.real
    return phi

def tracking_evolution(current_time,psi):
    # print(neighbour_track[-1])
    phi=phi_J_track(lat_track,current_time,J_target,neighbour_track[-1])
    # print(phi)
    # print(phi)
    # integrate static part of GPE
    psi_dot = -1j * (ham_onsite.dot(psi))
    psi_dot += -1j*np.exp(-1j*phi)*hop_left_op.dot(psi)
    psi_dot += -1j*np.exp(1j*phi)*hop_right_op.dot(psi)
    return psi_dot

phi_track=[]
neighbour_track=[]
J_track=[]
psi_t=psi_0
neighbour_track.append(hop_left_op.expt_value(psi_t))
phi_track.append(phi_J_track(lat_track,0,J_target,neighbour_track[-1]))
# current_partial=np.exp(-1j * phi_track[-1])*neighbour_track[-1]
# current = -1j * lat_track.a * lat_track.t* (current_partial - current_partial.conjugate())
J_track.append(J_target(0))
"""evolving system. For tracking things get a little sticky, since we want to read out expectations after each step"""

for newtime in tqdm(times[:-1]):
    # evolve forward by a single step. This is almost certainly not efficient, but is a first pass.
    """this version uses Quspin default, dop853"""
    psi_t = evolve(psi_t, newtime, np.array([newtime + delta]), tracking_evolution)
    """this version uses lsoda. It is unforgivably slow"""
    # solver_args=dict(method='bdf')
    # psi_t = evolve(psi_t, newtime, np.array([newtime + delta]), tracking_evolution,solver_name='lsoda')
    psi_t = np.squeeze(psi_t)
    """append expectations at each step"""
    neighbour_track.append(hop_left_op.expt_value(psi_t))
    phi_track.append(phi_J_track(lat_track,newtime+delta,J_target,neighbour_track[-1]))
    current_partial=lat_track.a * lat_track.t*np.exp(-1j * phi_track[-1])*neighbour_track[-1]
    current = -1j * (current_partial - current_partial.conjugate())
    J_track.append(current)

print("Evolution and expectation calculated! This one took {:.2f} seconds".format(time() - ti))
# Simple plot to check things are behaving.
plt.plot(times,phi_track)
plt.plot(times,phi_original)
plt.show()
J_track=np.array(J_track)
plt.plot(times,J_track.real/J_scale)
plt.plot(times,J_field,linestyle='--')
plt.show()
neighbour_track=np.array(neighbour_track)
plt.plot(times,neighbour_track.imag)
plt.show()
plt.plot(times,neighbour_track.real)
plt.show()
# psi_t = evolve(psi_0, 0, times, tracking_evolution)

outfile='./Data/expectations:{}sites-{}up-{}down-{}t0-{}U-{}cycles-{}steps-{}pbc'.format(L, N_up, N_down,
                                                                                                     t0, U, cycles,
                                                                                                     n_steps, pbc)
str=':U_t={:.2f}-a_scale={:.2f}-J_scale={:.2f}'.format(U_track,a_scale,J_scale)
expectations["tracking_current"]=J_track
expectations["tracking_phi"]=phi_track
expectations["tracking_neighbour"]=neighbour_track
print("Saving Expectations. We have {} of them".format(len(expectations)))
np.savez(outfile+str, **expectations)

print('All finished. Total time was {:.2f} seconds using {:d} threads'.format((time() - t_init), threads))

"""tests that things saved properly"""
# npzfile = np.load(outfile)
# print('npzfile.files: {}'.format(npzfile.files))
#
# current=npzfile["tracking_current"+str]
# old_current=npzfile["current"]
#
# plt.plot(times,current)
# plt.plot(times,old_current)
#
# # plt.plot(times,current_partial)
# plt.show()
#
# plt.plot(times,expectations["H"])
# plt.show()

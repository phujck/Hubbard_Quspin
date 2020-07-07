##########################################################
# Exact Lindblad type situation where chain is connected #
# via leads that inject fermions at the terminals. This  #
# takes inspiration from studies of diffusive transport  #
# For more detail see:          Phys. Rev. B 86, 125118  #
# QuSpin:  http://weinbe58.github.io/QuSpin/index.html   #
##########################################################
from __future__ import print_function, division

import os
import sys

"""Open MP and MKL should speed up the time required to run these simulations!"""
# threads = sys.argv[1]
threads = 16
os.environ['NUMEXPR_MAX_THREADS'] = '{}'.format(threads)
os.environ['NUMEXPR_NUM_THREADS'] = '{}'.format(threads)
os.environ['OMP_NUM_THREADS'] = '{}'.format(threads)
os.environ['MKL_NUM_THREADS'] = '{}'.format(threads)
# line 4 and line 5 below are for development purposes and can be remove
from quspin.operators import hamiltonian  # operators
from quspin.basis import spinful_fermion_basis_1d  # Hilbert space basis
from quspin.tools.evolution import evolve  #evolving system
from quspin.tools.measurements import obs_vs_time  # calculating dynamics
import numpy as np  # general math functions
from scipy.sparse.linalg import eigsh
from time import time  # tool for calculating computation time
import matplotlib.pyplot as plt  # plotting library
from quspin.tools.misc import get_matvec_function
import params
sys.path.append('../')
from tools import parameter_instantiate as hhg  # Used for scaling units.
import psutil
# note cpu_count for logical=False returns the wrong number for multi-socket CPUs.
print("logical cores available {}".format(psutil.cpu_count(logical=True)))
t_init = time()
np.__config__.show()
"""Hubbard model Parameters"""
SysParams=False #choose which set to take from sysparams
if SysParams:
    L = params.L # system size
    N_up = params.N_up  # number of fermions with spin up
    N_down = params.N_down # number of fermions with spin down
    # N_down = 0
    N = params.N # number of particles
    t0 = params.t0# hopping strength
    # U = 0*t0  # interaction strength
    U = params.U # interaction strength
    gamma = params.gamma
    mu = params.mu
    pbc = params.pbc

    """System Evolution Time"""
    t_max = params.t_max
    n_steps = params.n_steps
    times=params.times
    delta=params.delta
else:
    L = params.L2 # system size
    N_up = params.N_up2  # number of fermions with spin up
    N_down = params.N_down2 # number of fermions with spin down
    # N_down = 0
    N = params.N2 # number of particles
    t0 = params.t02# hopping strength
    # U = 0*t0  # interaction strength
    U = params.U2 # interaction strength
    gamma = params.gamma2
    mu = params.mu2
    pbc = params.pbc2
    t_max = params.t_max
    n_steps = params.n_steps
    times=params.times
    delta=params.delta

"""instantiate parameters with proper unit scaling"""
lat = hhg(nup=N_up, ndown=N_down, nx=L, ny=0, U=U, t=t0, pbc=pbc, gamma=gamma)
"""Define e^i*phi for later dynamics. Important point here is that for later implementations of tracking, we
will pass phi as a global variable that will be modified as appropriate during evolution"""
"""set up parameters for saving expectations later"""
outfile = './Data/Exact/expectations:{}sites-{}up-{}down-{}t0-{}U-{}t_max-{}steps-{}gamma-{}mu-{}pbc.npz'.format(L,
                                                                                                                  N_up,
                                                                                                                  N_down,
                                                                                                                  t0, U,
                                                                                                                  t_max,
                                                                                                                  n_steps,
                                                                                                                  gamma,
                                                                                                                  mu,
                                                                                                                  pbc)

"""create basis"""
# build spinful fermions basis. Note that the basis _cannot_ have number conservation as the leads inject and absorb
# fermions. This is frankly a massive pain, and the only gain we get
basis = spinful_fermion_basis_1d(L) #no symmetries
# basis = spinful_fermion_basis_1d(L, sblock=1)  # spin inversion symmetry
# basis = spinful_fermion_basis_1d(L,Nf=(N_up, N_down)) #number symmetry
# basis = spinful_fermion_basis_1d(L, Nf=(N_up, N_down),sblock=1) #parity and spin inversion symmetry
# basis = spinful_fermion_basis_1d(L, Nf=(N_up, N_down),a=1,kblock=1) #translation symmetry
print('Hilbert space size: {0:d}.\n'.format(basis.Ns))

"""building model"""
# define site-coupling lists
int_list = [[lat.U, i, i] for i in range(L)]  # onsite interaction

# create static lists
# Note that the pipe determines the spinfulness of the operator. | on the left corresponds to down spin, | on the right
# is for up spin. For the onsite interaction here, we have:


# add dynamic lists
hop_right = [[lat.t, i, i + 1] for i in range(L - 1)]  # hopping to the right OBC
hop_left = [[-lat.t, i, i + 1] for i in range(L - 1)]  # hopping to the left OBC

"""Add periodic boundaries"""
if lat.pbc:
    hop_right.append([lat.t, L - 1, 0])
    hop_left.append([-lat.t, L - 1, 0])

# After creating the site lists, we attach an operator and a time-dependent function to them
static_Hamiltonian_list = [
    ["n|n", int_list],  # onsite interaction
    ["+-|", hop_left],  # up hop left
    ["-+|", hop_right],  # up hop right
    ["|+-", hop_left],  # down hop left
    ["|-+", hop_right],  # down hop right
]
dynamic_Hamiltonian_list = []

"""build the Hamiltonian for actually evolving this bastard."""
# Hamiltonian builds an operator, the first argument is always the static operators, then the dynamic operators.
print(static_Hamiltonian_list)
H = hamiltonian(static_Hamiltonian_list, dynamic_Hamiltonian_list, basis=basis)
# print('Hamiltonian:\n',ham.toarray())
#
mat = H.toarray().real
fig, ax = plt.subplots()
# Using matshow here just because it sets the ticks up nicely. imshow is faster.
# ax.matshow(mat.real, cmap='seismic')
# plt.spy(mat,markersize=4,precision=10**(-6))
# plt.show()

"""build up the other operator expectations here as a dictionary"""
operator_dict = dict(H=H)
no_checks = dict(check_pcon=False, check_symm=False, check_herm=False)
# hopping operators for building current. Note that the easiest way to build an operator is just to cast it as an
# instance of the Hamiltonian class. Note in this instance the hops up and down have the e^iphi factor attached directly
operator_dict['neighbour'] = hamiltonian([["+-|", hop_left], ["|+-", hop_left]], [], basis=basis, **no_checks)
for j in range(L):
    # spin up densities for each site
    operator_dict["nup" + str(j)] = hamiltonian([["n|", [[1.0, j]]]], [], basis=basis, **no_checks)
    # spin down
    operator_dict["ndown" + str(j)] = hamiltonian([["|n", [[1.0, j]]]], [], basis=basis, **no_checks)
    # doublon densities
    operator_dict["D" + str(j)] = hamiltonian([["n|n", [[1.0, j, j]]]], [], basis=basis, **no_checks)

"""Lindblad operators. We'll have eight leads, for two spin species, two ends, and a pump and sink"""
# this version is slightly borked. Needs to be fixed.
# if lat.gamma:
#     print('I live!')
#     pos_rate = np.sqrt(gamma * (1 + mu))
#     neg_rate = np.sqrt(gamma * (1 - mu))
#     pos_start = [[pos_rate, 0]]
#     pos_end = [[pos_rate, L - 1]]
#     neg_start = [[neg_rate, 0]]
#     neg_end = [[neg_rate, L - 1]]
#     Lindblad_list = [
#         ["+|", neg_start],  # up injection left end
#         ["-|", pos_start],  # up absorption left end
#         ["|+", neg_start],  # down injection left end
#         ["|-", pos_start],  # down absorption left end
#         ["+|", pos_end],  # up injection right end
#         ["-|", neg_end],  # up absorption right end
#         ["|+", pos_end],  # down injection right end
#         ["|-", neg_end]  # down absorption right end
#     ]
#     # Build Lindbladian operators
#     L_op = hamiltonian(Lindblad_list, [], basis=basis, **no_checks)
#     L_dagger = L_op.getH()
#     L_daggerL = L_dagger * L_op
    # mat = L_op.toarray()
    # fig, ax = plt.subplots()
    # Using matshow here just because it sets the ticks up nicely. imshow is faster.
    # ax.matshow(mat.real, cmap='seismic')
    # plt.spy(mat,markersize=2,precision=10**(-6))
    # plt.show()

if lat.gamma:
    print('I live!')
    pos_rate = np.sqrt(gamma * (1 + mu))
    neg_rate = np.sqrt(gamma * (1 - mu))
    pos_start = [[pos_rate, 0]]
    pos_end = [[pos_rate, L - 1]]
    neg_start = [[neg_rate, 0]]
    neg_end = [[neg_rate, L - 1]]
    Lindblad_list = [
        hamiltonian([["+|", neg_start]],[], basis=basis, **no_checks),  # up injection left end
        hamiltonian([["-|", pos_start]], [],basis=basis, **no_checks),  # up absorption left end
        hamiltonian([["|+", neg_start]],[], basis=basis, **no_checks),  # down injection left end
        hamiltonian([["|-", pos_start]],[], basis=basis, **no_checks),  # down absorption left end
        hamiltonian([["+|", pos_end]], [], basis=basis, **no_checks), # up injection right end
        hamiltonian([["-|", neg_end]],[], basis=basis, **no_checks),  # up absorption right end
        hamiltonian([["|+", pos_end]],[], basis=basis, **no_checks),  # down injection right end
        hamiltonian([["|-", neg_end]],[], basis=basis, **no_checks)  # down absorption right end
    ]
    # Build Lindbladian operators
    # L_op = hamiltonian(Lindblad_list, [], basis=basis, **no_checks)
    # L_dagger_list =[L_op.getH() for L_op in Lindblad_list]
    L_daggerL_list =[L_op.getH() * L_op for L_op in Lindblad_list]

"""build ground state"""
print("calculating ground state")
# E, psi_0 = H.eigsh(k=1, which='SA')
# apparently you can get a speedup for the groundstate calculation using this method with multithread. Note that it's
# really not worth it unless you your number of sites gets _big_, and even then it's the time evolution which is going
# to kill you:
E, psi_0 = eigsh(H.aslinearoperator(time=0), k=1, which='SA')

print(type(psi_0))
print(psi_0.size)
# rho_0=np.outer(psi_0,psi_0)
rho_0=np.dot(psi_0,psi_0.T)
print(rho_0.shape)


print("ground state calculated, energy is {:.2f}".format(E[0]))
print('evolving system')
ti = time()
t_evo = ti

"""evolving system. In this simple case we'll just use the built in solver"""
#
#### determine the corresponding matvec routines ####
#
matvec = get_matvec_function(H.static)


#
# fast function (not as memory efficient)
def Lindblad_EOM_v2(time, rho, rho_out, rho_aux):
    """
    This function solves the complex-valued time-dependent GPE:
    $$ \dot\rho(t) = -i[H,\rho(t)] + 2\gamma\left( L\rho L^\dagger - \frac{1}{2}\{L^\dagger L, \rho \} \right) $$
    """
    rho = rho.reshape((H.Ns, H.Ns))  # reshape vector from ODE solver input
    ### Hamiltonian part
    # commutator term (unitary
    # rho_out = H._static.dot(rho))
    matvec(H.static, rho, out=rho_out, a=+1.0, overwrite_out=True)
    # rho_out -= (H._static.T.dot(rho.T)).T // RHS~rho.dot(H)
    matvec(H.static.T, rho.T, out=rho_out.T, a=-1.0, overwrite_out=False)
    # # no dynamic part to the Hamiltonian
    # for func, Hd in iteritems(H._dynamic):
    #     ft = func(time)
    #     # rho_out += ft*Hd.dot(rho)
    #     matvec(Hd, rho, out=rho_out, a=+ft, overwrite_out=False)
    #     # rho_out -= ft*(Hd.T.dot(rho.T)).T
    #     matvec(Hd.T, rho.T, out=rho_out.T, a=-ft, overwrite_out=False)
    # multiply by -i
    rho_out *= -1.0j
    #
    ### Lindbladian part (static only)
    # 1st Lindblad term (nonunitary)
    # rho_aux = 2*L.dot(rho)
    if lat.gamma:
        for L_op,L_daggerL in zip(Lindblad_list,L_daggerL_list):
            matvec(L_op.static, rho, out=rho_aux, a=+2.0, overwrite_out=True)
            # rho_out += (L.static.T.conj().dot(rho_aux.T)).T // RHS~rho_aux.dot(L_dagger)
            matvec(L_op.static.T.conj(), rho_aux.T, out=rho_out.T, a=+1.0, overwrite_out=False)
            # anticommutator (2nd Lindblad) term (nonunitary)
            # rho_out += gamma*L_daggerL._static.dot(rho)
            matvec(L_daggerL.static, rho, out=rho_out, a=-1.0, overwrite_out=False)
            # rho_out += gamma*(L_daggerL._static.T.dot(rho.T)).T // RHS~rho.dot(L_daggerL)
            matvec(L_daggerL.static.T, rho.T, out=rho_out.T, a=-1.0, overwrite_out=False)
    #
    return rho_out.ravel()  # ODE solver accepts vectors only


#
# define auxiliary arguments
EOM_args = (np.zeros((H.Ns, H.Ns), dtype=np.complex128, order="C"),  # auxiliary variable rho_out
            np.zeros((H.Ns, H.Ns), dtype=np.complex128, order="C"))  # auxiliary variable rho_aux
#
##### time-evolve state according to Lindlad equation
# define real time vector

# define initial state
# rho0 = np.array([[0.5, 0.5j], [-0.5j, 0.5]], dtype=np.complex128)
#
# evolution
rho_t = evolve(rho_0, times[0], times, Lindblad_EOM_v2, f_params=EOM_args, iterate=False, atol=1E-12, rtol=1E-12,verbose=True)
print(rho_t.shape)
# this version returns the generator for psi
# psi_t=ham.evolve(psi_0,0.0,times,iterate=True)

# this version returns psi directly, last dimension contains time dynamics. The squeeze is necessary for the
# obs_vs_time to work properly
# psi_t = H.evolve(psi_0, 0.0, times)
# psi_t = np.squeeze(psi_t)
print("Evolution done! This one took {:.2f} seconds".format(time() - ti))
# calculate the expectations for every bastard in the operator dictionary
ti = time()
# note that here the expectations
expectations = obs_vs_time(rho_t, times, operator_dict)
print(type(expectations))
current_partial = (expectations['neighbour'])
current = 1j * lat.a * (current_partial - current_partial.conjugate())
expectations['current'] = current
print("Expectations calculated! This took {:.2f} seconds".format(time() - ti))
expectations['evotime'] = time() - t_evo

print("Saving Expectations. We have {} of them".format(len(expectations)))
np.savez(outfile, **expectations)
#
#

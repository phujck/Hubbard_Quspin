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
threads = 40
os.environ['NUMEXPR_MAX_THREADS'] = '{}'.format(threads)
os.environ['NUMEXPR_NUM_THREADS'] = '{}'.format(threads)
os.environ['OMP_NUM_THREADS'] = '{}'.format(threads)
os.environ['MKL_NUM_THREADS'] = '{}'.format(threads)
# line 4 and line 5 below are for development purposes and can be remove
from quspin.operators import hamiltonian, exp_op  # operators
from quspin.basis import spinless_fermion_basis_1d  # Hilbert space basis
from quspin.tools.evolution import expm_multiply_parallel  #evolving system
from quspin.tools.measurements import obs_vs_time  # calculating dynamics
import numpy as np  # general math functions
from scipy import sparse
from scipy.sparse.linalg import eigsh
from time import time  # tool for calculating computation time
import matplotlib.pyplot as plt  # plotting library
from quspin.tools.misc import get_matvec_function
import params
sys.path.append('../')
from tools import parameter_instantiate as hhg  # Used for scaling units.
import psutil
from tqdm import tqdm
import timeit
from numba import jit,njit
import collections, functools, operator
# note cpu_count for logical=False returns the wrong number for multi-socket CPUs.
print("logical cores available {}".format(psutil.cpu_count(logical=True)))
t_init = time()
np.__config__.show()
"""Hubbard model Parameters"""
SysParams=True #choose which set to take from sysparams
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
    rank=params.rank

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
    rank=params.rank
"""instantiate parameters with proper unit scaling"""
lat = hhg(nup=N_up, ndown=N_down, nx=L, ny=0, U=U, t=t0, pbc=pbc, gamma=gamma,mu=mu)
outfile = './Data/Approx/expectations:{}sites-{}up-{}down-{}t0-{}U-{}t_max-{}steps-{}gamma-{}mu-{}rank-{}pbc.npz'.format(L,
                                                                                                                  N_up,
                                                                                                                  N_down,
                                                                                                                  t0, U,
                                                                                                                  t_max,
                                                                                                                  n_steps,
                                                                                                                  gamma,
                                                                                                                  mu,rank,
                                                                                                                  pbc)
lat.gamma=2*lat.gamma
"""create basis"""
# build spinful fermions basis. Note that the basis _cannot_ have number conservation as the leads inject and absorb
# fermions. This is frankly a massive pain, and the only gain we get
basis = spinless_fermion_basis_1d(L) #no symmetries
# basis = spinful_fermion_basis_1d(L, sblock=1)  # spin inversion symmetry
# basis = spinful_fermion_basis_1d(L,Nf=(N_up, N_down)) #number symmetry
# basis = spinful_fermion_basis_1d(L, Nf=(N_up, N_down),sblock=1) #parity and spin inversion symmetry
# basis = spinful_fermion_basis_1d(L, Nf=(N_up, N_down),a=1,kblock=1) #translation symmetry
print('Hilbert space size: {0:d}.\n'.format(basis.Ns))

"""building model"""
# define site-coupling lists

# create static lists
# Note that the pipe determines the spinfulness of the operator. | on the left corresponds to down spin, | on the right
# is for up spin. For the onsite interaction here, we have:


# add dynamic lists
hop_left = [[-lat.t, i, i + 1] for i in range(L - 1)]  # hopping to the left OBC
hop_right = [[lat.t, i, i + 1] for i in range(L - 1)]  # hopping to the left OBC

"""Add periodic boundaries"""
if lat.pbc:
    hop_left.append([-lat.t, L - 1, 0])

# After creating the site lists, we attach an operator and a time-dependent function to them
static_Hamiltonian_list = [
    ["+-", hop_left],  # up hop left
    ["-+", hop_right],  # down hop left
]
dynamic_Hamiltonian_list = []

"""build the Hamiltonian for actually evolving this bastard."""
# Hamiltonian builds an operator, the first argument is always the static operators, then the dynamic operators.
print(static_Hamiltonian_list)
H = hamiltonian(static_Hamiltonian_list, dynamic_Hamiltonian_list, basis=basis)
# print('Hamiltonian:\n',ham.toarray())
#
# mat = H.toarray().real
# fig, ax = plt.subplots()
# Using matshow here just because it sets the ticks up nicely. imshow is faster.
# ax.matshow(mat.real, cmap='seismic')
# plt.spy(mat,markersize=4,precision=10**(-6))
# plt.show()

"""build up the other operator expectations here as a dictionary"""
operator_dict = dict(H=H)
no_checks = dict(check_pcon=False, check_symm=False, check_herm=False)
# hopping operators for building current. Note that the easiest way to build an operator is just to cast it as an
# instance of the Hamiltonian class. Note in this instance the hops up and down have the e^iphi factor attached directly
operator_dict['neighbour'] = hamiltonian([["+-", hop_left]], [], basis=basis, **no_checks)
for j in range(L):
    # spin up densities for each site
    operator_dict["n" + str(j)] = hamiltonian([["n", [[1.0, j]]]], [], basis=basis, **no_checks)
    # spin down
    # doublon densities
"""Lindblad operators. We'll have eight leads, for two spin species, two ends, and a pump and sink"""
if lat.gamma:
    print('I live!')
    pos_rate = np.sqrt(lat.gamma * (1 + lat.mu))
    neg_rate = np.sqrt(lat.gamma * (1 - lat.mu))
    pos_start = [[pos_rate, 0]]
    pos_end = [[pos_rate, L - 1]]
    neg_start = [[neg_rate, 0]]
    neg_end = [[neg_rate, L - 1]]
    Lindblad_list = [
        hamiltonian([["+", neg_start]],[], basis=basis, **no_checks),  # up injection left end
        hamiltonian([["-", pos_start]], [],basis=basis, **no_checks),  # up absorption left end
        hamiltonian([["+", pos_end]], [], basis=basis, **no_checks), # up injection right end
        hamiltonian([["-", neg_end]],[], basis=basis, **no_checks)  # down absorption right end
    ]
else:
    Lindblad_list=[]
K = len(Lindblad_list)
if K:
    prefactor = np.sqrt(1 / (2 * K))
else:
    prefactor=1
def unitaries(H, c_op_list, dt,K):

    U_ops = []
    if K==0:
        U_ops.append(expm_multiply_parallel(H.tocsr(),a=-1j*delta,dtype=np.complex128))
    else:
        for op in c_op_list:
            exponent = -1j * H * dt
            exponent += (dt * K / 2) * (op * op - op.getH() * op)
            extra = 1j * np.sqrt(K * dt) * op
            P1 = exponent + extra
            P2 = exponent - extra
            # P1array=P1.toarray()
            # P2array=P2.toarray()
            # P1array=P1array-P2array
            # plt.spy(P1array.real,markersize=2,precision=10**(-6))
            # plt.show()
            # plt.spy(P1array.imag, markersize=2, precision=10 ** (-6))
            # plt.show()

            U_prop=expm_multiply_parallel(P1.tocsr(),a=1,dtype=np.complex128)
            V_prop=expm_multiply_parallel(P2.tocsr(),a=1,dtype=np.complex128)
            # U_prop = exp_op(P1, a=1)
            # V_prop = exp_op(P2, a=1)
            U_ops.append(U_prop)
            U_ops.append(V_prop)
    return U_ops

L_ops=unitaries(H,Lindblad_list,delta,K)
print(len(L_ops))
print('Unitaries Constructed!')

"""build ground state"""
print("calculating ground state")
E, psi_0 = H.eigsh(k=1, which='SA')
# apparently you can get a speedup for the groundstate calculation using this method with multithread. Note that it's
# really not worth it unless you your number of sites gets _big_, and even then it's the time evolution which is going
# to kill you:
# E, psi_0 = eigsh(H.aslinearoperator(time=0), k=1, which='SA')

print(type(psi_0))
print(psi_0.size)



print("ground state calculated, energy is {:.2f}".format(E[0]))
print('evolving system')
ti = time()
t_evo = ti

"""Set-up for evolving unitaries"""
def overlap(psi_list):
    matrix = np.zeros((len(psi_list), len(psi_list)), dtype=np.complex_)
    i=0
    for psi in psi_list:
        j = 0
        for psi2 in psi_list:
            # print(np.allclose(psi, psi2))
            overlap=np.vdot(psi, psi2)
            # print(overlap)
            matrix[i,j]=overlap
            j+=1
        i += 1

    # print(matrix)
    return matrix

@njit
def overlap_quick(psi_list):
    matrix = psi_list.conj()@psi_list.T
    # print(matrix)
    return matrix

@njit
def orthogonalise(new_psi):
    # newtime=time()
    # mat = overlap(new_psis)
    # print('old method takes {} seconds'.format(time()-newtime))
    # newtime=time()
    mat = new_psi.conj() @ new_psi.T
    # mat=overlap_quick(np.array(new_psi))
    # print('new method takes {} seconds'.format(time()-newtime))
    w, v = np.linalg.eigh(mat)
    v = v[:, ::-1]
    # if len(new_psi)>rank:
    #     print('using sparse method')
        # w, v = eigsh(mat,k=rank, which='LM')
    # else:
    #     w, v = np.linalg.eigh(mat)
    #     v = v[:, ::-1]
    return v.T

def orthogonal_psis(psi_list,v,rank):
    if len(psi_list) < rank:
        newrank = len(psi_list)
    else:
        newrank = rank
    g_list=[]
    norm = 0
    for i in range(newrank):
        # print(i)
        g = np.zeros(len(psi_list[0]),dtype='complex128')
        for j,psi in enumerate(psi_list):
            # print(U[i,j])
            # print(psi_list[j].dims)
            g_unit = v[i, j] * psi
            g += g_unit
        norm+=np.vdot(g,g)
        # norm=1
        g_list.append(g)
    g_list=[g/np.sqrt(norm) for g in g_list]
    return g_list


# def orthogonal_psis(psi_list,v):
#     g_list=[]
#     norm=0
#     for i in range(len(psi_list)):
#         # print(i)
#         g = np.zeros(len(psi_list[0,:]),dtype='complex128')
#         for j in range(len(psi_list)):
#             # print(U[i,j])
#             # print(psi_list[j].dims)
#             g_unit = v[i, j] * psi_list[j,:]
#             g += g_unit
#         norm += np.vdot(g, g)
#         # norm=1
#         g_list.append(g)
#     g_list=[g/np.sqrt(norm) for g in g_list]
#     return g_list


# def one_step_alt(psis,U_ops,rank):
#     new_psis = []
#     for psi in psis:
#         psi = psi.flatten()
#         for U in U_ops:
#             # print('pre-unitary')
#             # print(np.vdot(psi, psi))
#             newpsi = prefactor*U.dot(psi)
#             # U.dot(newpsi,overwrite_v=True)
#             # newpsi=newpsi/np.sqrt(np.vdot(newpsi,newpsi)/prefactor)
#             # print('post-unitary')
#             # print(np.vdot(newpsi, newpsi))
#             new_psis.append(newpsi)
#         # print('now have {} wavefunction'.format(len(new_psis)))
#         # print(np.array(new_psis).shape)
#     new_psis = orthogonalise(new_psis, rank)
#     new_psis=np.sum(np.array(new_psis),axis=0)
#
#     # if len(new_psis) > rank:
#     #     new_psis = orthogonalise(new_psis, rank)
#     return new_psis


def one_step(psis,U_ops,rank):
    new_psis = []
    for psi in psis:
        psi = psi.flatten()
        for U in U_ops:
            # print('pre-unitary')
            # print(np.vdot(psi, psi))
            newpsi = U.dot(psi)
            # U.dot(newpsi,overwrite_v=True)
            # newpsi=newpsi/np.sqrt(np.vdot(newpsi,newpsi)/prefactor)
            # print('post-unitary')
            # print(np.vdot(newpsi, newpsi))
            new_psis.append(newpsi)
        # print('now have {} wavefunction'.format(len(new_psis)))
        # print(np.array(new_psis).shape)
    # new_psis = orthogonalise(new_psis, rank)
    # new_psis=np.sum(np.array(new_psis),axis=0)

    # if len(new_psis) > rank:
    #     new_psis = orthogonalise(new_psis, rank)
    if len(new_psis) <rank:
        zeros=np.zeros(len(new_psis[0]))
        while len(new_psis)<rank:
            new_psis.append(zeros)
    # new_psis = orthogonalise(new_psis, rank)
    # print(np.array(new_psis,dtype='complex128').size)
    # new_psis=np.array(new_psis)
    U_mat = orthogonalise(np.array(new_psis))
    new_psis=orthogonal_psis(new_psis,U_mat,rank)
    # print(len(np.array(new_psis,dtype='complex128')))


    return new_psis

zeropsi=np.zeros(len(psi_0),dtype='complex128')
psis=[psi_0]
time_psis=[]
print(psi_0.shape)
for j in tqdm(range(n_steps-1)):
    # s_rank.append(expectations(psis,sz_list+sx_list))
    psis=one_step(psis,L_ops,rank)
    # print(np.array(psis).shape)
    # print(psi_0.shape)
    time_psis.append(np.array(psis))
    # norm=np.sum(norms)
    # norms = [np.vdot(psi, psi) for psi in psis]
    # norm = np.sum(norms)
    # print(norm)
    # print(norm)
    # psis = [psi/np.sqrt(norm) for psi in psis]
    # mat = overlap(psis)
    # # print(np.sum(np.diag(mat)))
    # fig, ax = plt.subplots()
    # ax.matshow(mat.real, cmap='seismic')
    #
    # for (i, j), z in np.ndenumerate(mat.real):
    #     ax.text(j, i, '{:0.5f}'.format(z), ha='center', va='center')
    #
    #
    # plt.show()

final_psis=np.array(time_psis).T
print(final_psis.shape)

# final_psis=np.sum(final_psis,axis=1)
final_psis=np.swapaxes(final_psis,1,2)
print('Low rank propagation done. Time taken %.5f seconds' % (time()-ti))

if K==0:
    psi_t=final_psis[:,:,0]
    expectations = obs_vs_time(psi_t, times, operator_dict)
    current_partial = (expectations['neighbour'])
    current = 1j * lat.a * (current_partial - current_partial.conjugate())
    expectations['current'] = current
else:
    expectation_dics = []
    for j in range(rank):
        psi_t=(final_psis[:,:,j])
        # print(psi_0.shape)
        # psi_t=np.concatenate(psi_0,psi_t,axis=1)
        # print(psi_t.shape)
        new=np.hstack((psi_0/np.sqrt(rank),psi_t))
        psi_t=new
        # print(psi_t.shape)
        expectations = obs_vs_time(psi_t, times, operator_dict)
        print(type(expectations))
        current_partial = (expectations['neighbour'])
        current = 1j * lat.a * (current_partial - current_partial.conjugate())
        expectations['current'] = current
        expectation_dics.append(expectations)
    final_expectations={}
    for d in expectation_dics:
        for k in d.keys():
            final_expectations[k] = final_expectations.get(k, 0) + d[k]

    expectations=final_expectations
"""alternate method"""
# psis=psi_0.flatten()
# time_psis=[]
# for j in tqdm(range(n_steps)):
#     # s_rank.append(expectations(psis,sz_list+sx_list))
#     psis=one_step_alt([psis],L_ops,rank)
#     # print(psis.shape)
#     # print(psi_0.shape)
#     time_psis.append(psis)
#     # norm=np.sum(norms)
#     # print(norm)
#     # psis = [psi/np.sqrt(norm) for psi in psis]
#     # mat = overlap(psis)
#     # # print(np.sum(np.diag(mat)))
#     # fig, ax = plt.subplots()
#     # ax.matshow(mat.real, cmap='seismic')
#     #
#     # for (i, j), z in np.ndenumerate(mat.real):
#     #     ax.text(j, i, '{:0.5f}'.format(z), ha='center', va='center')
#     #
#     #
#     # plt.show()
#
# print(len(time_psis[0]))
# psi_t=np.array(time_psis).T
# print(psi_t.shape)
# expectations = obs_vs_time(psi_t, times, operator_dict)
# current_partial = (expectations['neighbour'])
# current = 1j * lat.a * (current_partial - current_partial.conjugate())
# expectations['current'] = current

#
#
# # for j in tqdm(range(n_steps)):
# #     # s_rank.append(expectations(psis,sz_list+sx_list))
# #     psis=one_step_alt(psis,L_ops,zeropsi,rank)
# #     time_psis.append(psis)
# #     # norms=[np.vdot(psi,psi) for psi in psis]
# #     # norm=np.sum(norms)
# #     # psis = [psi/np.sqrt(norm) for psi in psis]
# #
# # psi_t=np.array(time_psis).T
# # print(psi_t.shape)
print('Low rank propagation done. Time taken %.5f seconds' % (time()-ti))


#
#
#
#
# # psi_list = [psi for psi in psis]
# # this version returns the generator for psi
# # psi_t=ham.evolve(psi_0,0.0,times,iterate=True)
#
# # this version returns psi directly, last dimension contains time dynamics. The squeeze is necessary for the
# # obs_vs_time to work properly
# # psi_t = H.evolve(psi_0, 0.0, times)
# # psi_t = np.squeeze(psi_t)
# print("Evolution done! This one took {:.2f} seconds".format(time() - ti))
# # calculate the expectations for every bastard in the operator dictionary
# ti = time()
# # note that here the expectations
#
print(current.shape)
print("Expectations calculated! This took {:.2f} seconds".format(time() - ti))
expectations['evotime'] = time() - t_evo

print("Saving Expectations. We have {} of them".format(len(expectations)))
np.savez(outfile, **expectations)
#


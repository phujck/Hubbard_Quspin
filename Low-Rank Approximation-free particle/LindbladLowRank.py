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
threads = 20
os.environ['NUMEXPR_MAX_THREADS'] = '{}'.format(threads)
os.environ['NUMEXPR_NUM_THREADS'] = '{}'.format(threads)
os.environ['OMP_NUM_THREADS'] = '{}'.format(threads)
os.environ['MKL_NUM_THREADS'] = '{}'.format(threads)
# line 4 and line 5 below are for development purposes and can be remove
from quspin.operators import hamiltonian  # operators
from quspin.basis import spinful_fermion_basis_1d  # Hilbert space basis
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
if lat.gamma:
    print('I live!')
    pos_rate = np.sqrt(lat.gamma * (1 + lat.mu))
    neg_rate = np.sqrt(lat.gamma * (1 - lat.mu))
    pos_start = [[pos_rate, 0]]
    pos_end = [[pos_rate, L - 1]]
    neg_start = [[neg_rate, 0]]
    neg_end = [[neg_rate, L - 1]]
    Lindblad_list = [
        hamiltonian([["+|", neg_start]],[], basis=basis, **no_checks),  # up injection left end
        # hamiltonian([["-|", pos_start]], [],basis=basis, **no_checks),  # up absorption left end
        # hamiltonian([["|+", neg_start]],[], basis=basis, **no_checks),  # down injection left end
        # hamiltonian([["|-", pos_start]],[], basis=basis, **no_checks),  # down absorption left end
        # hamiltonian([["+|", pos_end]], [], basis=basis, **no_checks), # up injection right end
        # hamiltonian([["-|", neg_end]],[], basis=basis, **no_checks),  # up absorption right end
        # hamiltonian([["|+", pos_end]],[], basis=basis, **no_checks),  # down injection right end
        hamiltonian([["|-", neg_end]],[], basis=basis, **no_checks)  # down absorption right end
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
        U_ops.append(expm_multiply_parallel(H.tocsr(),a=-1j*dt,dtype=np.complex128))
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
            # U_prop = expm_multiply_parallel(P1, a=1)
            # V_prop = expm_multiply_parallel(P2, a=1)
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
    for i in range(len(psi_list)):
        for j in range(i, len(psi_list)):
            # overlap=psi_list[i].dag()*psi_list[j]
            # print('method 1')
            # print(overlap.data)
            # print('method 2')
            o2 = np.vdot(psi_list[i],psi_list[j])
            print(o2)
            # o2 = np.vdot(psi_list[j], psi_list[i])
            # print(o2)
            matrix[i, j] = o2
            if j > i:
                matrix[j, i] = np.conj(o2)
    # print(matrix)
    return matrix


# def orthogonalise(psi_list, U, zeroobj, rank):
#     g_list = []
#     if rank > len(psi_list):
#         rank = len(psi_list)
#     U = U[:, :rank - 1].T
#     norm = 0
#     for psi in psi_list:
#         g=np.dot(U,psi)
#         g_list.append(g)
#     for g in g_list:
#         for k in g_list:
#             print('overlaps')
#             print(np.vdot(g,k))
#         # g_list.append(g)
#     g_list=[g/np.sqrt(norm) for g in g_list]
#     #Try adding the orthonormalised vectors back together in this version
#     return g_list

def orthogonalise(psi_list, U, zeroobj, rank):
    g_list = []
    U = U.T
    if rank > len(psi_list):
        rank = len(psi_list)
    norm = 0
    for i in range(rank):
        g = zeroobj
        for j in range(len(psi_list)):
            # print(U[i,j])
            # print(psi_list[j].dims)
            g_unit = U[i, j] * psi_list[j]
            g += g_unit
        norm+=np.vdot(g,g)
        # norm=1
        g_list.append(g)
    g_list=[g/np.sqrt(norm) for g in g_list]

    for g in g_list:
        for k in g_list:
            print('overlaps')
            print(np.vdot(g,k))
            print(np.allclose(g,k))
    mat = overlap(g_list)
    # print(np.sum(np.diag(mat)))
    fig, ax = plt.subplots()
    ax.matshow(mat.real, cmap='seismic')

    for (i, j), z in np.ndenumerate(mat.real):
        ax.text(j, i, '{:0.5f}'.format(z), ha='center', va='center')

    # plt.spy(mat,markersize=2,precision=10**(-6))
    plt.show()
    return g_list

def one_step(psis,U_ops,zeropsi,rank):
    new_psis=[]
    for psi in psis:
        psi=psi.flatten()
        for U in U_ops:
            print('pre-unitary')
            print(np.vdot(psi,psi))
            newpsi=U.dot(psi)
            # newpsi=newpsi/np.sqrt(np.vdot(newpsi,newpsi)/prefactor)
            print('post-unitary')
            print(np.vdot(newpsi,newpsi))
            new_psis.append(newpsi)
    print('orthogonality check')
    # for psi in new_psis:
    #     for psi2 in new_psis:
    #         print(np.allclose(psi,psi2))
    #         print(np.vdot(psi,psi2))
    if len(new_psis) > rank:
        mat = overlap(new_psis)
        # w, v = np.linalg.eig(mat)
        # idx = w.argsort()[::-1]
        # w = w[idx]
        w,v=np.linalg.eigh(mat)
        w = w[::-1]
        v = v[:, ::-1]
        print('eigenvalues')
        print(w)
        # print(v)
        # w, v2 = sparse.linalg.eigsh(mat, k=rank, which='LM')
        # print(w)
        # v2=np.flip(v2,axis=1)
        # newmat= v.conj().T@mat@v
        # mat=newmat
        # print(np.sum(np.diag(mat)))
        # fig, ax = plt.subplots()
        # ax.matshow(mat.real, cmap='seismic')
        #
        # for (i, j), z in np.ndenumerate(mat.real):
        #     ax.text(j, i, '{:0.5f}'.format(z), ha='center', va='center')
        # # print(np.sum(np.diag(mat)))
        # fig, ax = plt.subplots()
        # ax.matshow(mat.real, cmap='seismic')



        # for j in range(len(w)):
        #     print(np.vdot(v[:,0],v[:,j]))

        new_psis = orthogonalise(new_psis, v, zeropsi, rank)
    return new_psis

def one_step_alt(psis,U_ops,zeropsi,rank):
    new_psis=[]
    for psi in psis:
        psi=psi.flatten()
        for U in U_ops:
            newpsi=prefactor*U.dot(psi)
            new_psis.append(newpsi)
    if len(new_psis) > rank:
        mat = overlap(new_psis)
        # w, v = np.linalg.eig(mat)
        # w = w[idx]
        # v = v[:, idx]
        # w,v=np.linalg.eigh(mat)
        w,v=np.linalg.eigsh(mat,k=rank,which='LM')
        print(w)
        new_psis = np.array(orthogonalise(new_psis, v, zeropsi, rank))
        print(new_psis.shape)
        new_psis=np.sum(new_psis,axis=0)
        print(new_psis.shape)
    return [new_psis]
#
# fast function (not as memory efficient)


#
# define auxiliary arguments

##### time-evolve state according to Lindlad equation
# define real time vector

# define initial state
# rho0 = np.array([[0.5, 0.5j], [-0.5j, 0.5]], dtype=np.complex128)
#
# evolution
zeropsi=np.zeros(H.Ns,dtype='complex128')
psis=[psi_0]
time_psis=[psi_0]


for j in tqdm(range(n_steps)):
    # s_rank.append(expectations(psis,sz_list+sx_list))
    psis=one_step(psis,L_ops,zeropsi,rank)
    time_psis.append(psis)
    norms=[np.vdot(psi,psi) for psi in psis]
    norm=np.sum(norms)
    psis = [psi/np.sqrt(norm) for psi in psis]
    # mat = overlap(psis)
    # # print(np.sum(np.diag(mat)))
    # fig, ax = plt.subplots()
    # ax.matshow(mat.real, cmap='seismic')
    #
    # for (i, j), z in np.ndenumerate(mat.real):
    #     ax.text(j, i, '{:0.5f}'.format(z), ha='center', va='center')
    #
    #
    # # plt.spy(mat,markersize=2,precision=10**(-6))
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
        psi_t=final_psis[:,:,j]
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


# for j in tqdm(range(n_steps)):
#     # s_rank.append(expectations(psis,sz_list+sx_list))
#     psis=one_step_alt(psis,L_ops,zeropsi,rank)
#     time_psis.append(psis)
#     # norms=[np.vdot(psi,psi) for psi in psis]
#     # norm=np.sum(norms)
#     # psis = [psi/np.sqrt(norm) for psi in psis]
#
# psi_t=np.array(time_psis).T
# print(psi_t.shape)
# print('Low rank propagation done. Time taken %.5f seconds' % (time()-ti))
#
#
# expectations = obs_vs_time(psi_t, times, operator_dict)
# current_partial = (expectations['neighbour'])
# current = 1j * lat.a * (current_partial - current_partial.conjugate())
# expectations['current'] = current




# psi_list = [psi for psi in psis]
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

print(current.shape)
print("Expectations calculated! This took {:.2f} seconds".format(time() - ti))
expectations['evotime'] = time() - t_evo

print("Saving Expectations. We have {} of them".format(len(expectations)))
np.savez(outfile, **expectations)
#
#

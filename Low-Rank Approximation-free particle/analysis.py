import sys
import numpy as np
import itertools
import matplotlib.pyplot as plt
import numpy.ma as ma
from matplotlib import cm as cm
from scipy.signal import blackman
from scipy.signal import stft
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from scipy import signal
from scipy import ndimage
from params import *

sys.path.append('../')
from tools import parameter_instantiate as hhg  # Used for scaling units.


def spectrum_welch(at, delta):
    return signal.welch(at, 1. / delta, nperseg=len(at), scaling='spectrum')


def spectrum(at, delta):
    w = np.fft.rfftfreq(len(at), delta)
    spec = 2. * (abs(np.fft.rfft(at)) ** 2.) / (len(at)) ** 2.
    return (w, spec)


def spectrum_hanning(at, delta):
    w = np.fft.rfftfreq(len(at), delta)
    win = np.hanning(len(at))
    spec = 2. * (abs(np.fft.rfft(win * at)) ** 2.) / (sum(win ** 2.)) ** 2
    return (w, spec)


def plot_spectra(U, w, spec, min_spec, max_harm):
    # spec = np.log10(spec)
    xlines = [2 * i - 1 for i in range(1, 6)]
    for i, j in enumerate(U):
        plt.semilogy(w, spec[:, i], label='$\\frac{U}{t_0}=$ %.1f' % (j))
        axes = plt.gca()
        axes.set_xlim([0, max_harm])
        axes.set_ylim([10 ** (-15), spec.max()])
    for xc in xlines:
        plt.axvline(x=xc, color='black', linestyle='dashed')
        plt.xlabel('Harmonic Order')
        plt.ylabel('HHG spectra')
    plt.legend(loc='upper right')
    plt.show()



params = {
    'axes.labelsize': 40,
    # 'legend.fontsize': 28,
    'legend.fontsize': 28,
    'xtick.labelsize': 28,
    'ytick.labelsize': 28,
    'figure.figsize': [20, 12],
    'text.usetex': True,
    'lines.linewidth' : 3,
    'lines.markersize' : 15
}
plt.rcParams.update(params)
# print(plt.rcParams.keys())

"""Hubbard model Parameters"""
"""instantiate parameters with proper unit scaling"""
lat = hhg(nup=N_up, ndown=N_down, nx=L, U=U, t=t0, pbc=pbc, gamma=gamma, mu=mu)
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

# outfile2 = './Data/Exact/expectations:{}sites-{}up-{}down-{}t0-{}U-{}t_max-{}steps-{}gamma-{}mu-{}pbc.npz'.format(L2,
#                                                                                                                  N_up2,
#                                                                                                                  N_down2,
#                                                                                                                  t02, U2,
#                                                                                                                  t_max2,
#                                                                                                                  n_steps2,
#                                                                                                                  gamma2,
#                                                                                                                  mu2,
#                                                                                                                  pbc)

outfile = './Data/Approx/expectations:{}sites-{}up-{}down-{}t0-{}U-{}t_max-{}steps-{}gamma-{}mu-{}rank-{}pbc.npz'.format(
    L2,
    N_up2,
    N_down2,
    t0, U,
    t_max,
    n_steps2,
    gamma,
    mu, rank2,
    pbc)

outfile2 = './Data/Approx/expectations:{}sites-{}up-{}down-{}t0-{}U-{}t_max-{}steps-{}gamma-{}mu-{}rank-{}pbc.npz'.format(
    L2,
    N_up2,
    N_down2,
    t0, U,
    t_max,
    n_steps2,
    gamma,
    mu, rank,
    pbc)
figparams = '{}sites-{}up-{}down-{}t0-{}U-{}t_max-{}steps-{}gamma-{}mu-{}pbc.pdf'.format(L,
                                                                                         N_up,
                                                                                         N_down,
                                                                                         t0, U,
                                                                                         t_max,
                                                                                         n_steps,
                                                                                         gamma,
                                                                                         mu,
                                                                                         pbc)
expectations = np.load(outfile)
# expectations2 = expectations
expectations2 = np.load(outfile2)
sites = [j for j in range(L)]
sites2 = sites
"""show the expectations available here"""
print('expectations available: {}'.format(expectations.files))

# """load expectations. For densities, time is first dimension, site number is second"""
J_field = expectations['current']
# up_densities = np.vstack([expectations["n" + str(j)] for j in range(L)]).T
#
#
J_field2 = expectations2['current']
# up_densities2 = np.vstack([expectations2["n" + str(j)] for j in range(L2)]).T
#
# print(up_densities.shape)

plt.subplot(211)
plt.plot(times2, J_field2,label='low-rank',color='red')
plt.plot(times,J_field,linestyle='--',label='exact',color='black')
# plt.plot(times2, ndimage.gaussian_filter(J_field2.real,1,1),label='low-rank',color='red')
# plt.plot(times, ndimage.gaussian_filter(J_field.real,1,1),linestyle='--',label='exact',color='black')
plt.xlabel('Time')
plt.ylabel('$J(t)$')
plt.legend(loc='upper left')
# plt.annotate('a)', xy=(0.3, np.max(J_field) - 0.08), fontsize=25)
prev_max=0
plt.subplot(212)
method = 'welch'
min_spec = 15
max_harm = 10
gabor = 'fL'
exact = np.gradient(J_field, delta)
exact2 = np.gradient(J_field2, delta2)
w, spec = spectrum_welch(exact.real, delta)
w2, spec2 = spectrum_welch(exact2.real, delta2)

w *= 2. * np.pi
w2 *= 2. * np.pi
max_value = max(spec)
max_index = np.where(spec == max_value)
w_scale = w[max_index]
print(w_scale)
w = w / w_scale
max_value = max(spec2)
max_index = np.where(spec2 == max_value)
w_scale = w2[max_index]
print(w_scale)
w2=w2/w_scale
plt.semilogy(w, spec, linestyle='--',color='black')
plt.semilogy(w2, spec2,color='red')
axes = plt.gca()
# axes.set_xlim([0, max_harm])
if spec.max() > prev_max:
    prev_max = spec.max() * 5
# axes.set_ylim([10 ** (-min_spec), prev_max])
# xlines = [2 * i - 1 for i in range(1, 6)]
xlines = [ i  for i in range(1, 11)]
for xc in xlines:
    plt.axvline(x=xc, color='black', linestyle='dashed')
plt.xlabel('$\\omega$')
plt.ylabel('HHG spectra')
# plt.legend(loc='upper right')
plt.show()
print(expectations2['evotime'])

# plt.subplot(211)
# plt.plot(times, J_field,linestyle='--',label='exact',color='black')
# plt.xlabel('Time')
# plt.ylabel('$J(t)$')
# plt.legend(loc='upper left')
# # plt.annotate('a)', xy=(0.3, np.max(J_field) - 0.08), fontsize=25)
# prev_max=0
# plt.subplot(212)
# method = 'welch'
# min_spec = 18
# max_harm = 10
# gabor = 'fL'
# exact = np.gradient(J_field, delta)
# w, spec = spectrum_welch(exact.real, delta)
# w *= 2. * np.pi
# plt.semilogy(w, spec, linestyle='--',color='black')
# axes = plt.gca()
# axes.set_xlim([0, max_harm])
# if spec.max() > prev_max:
#     prev_max = spec.max() * 5
# axes.set_ylim([10 ** (-min_spec), prev_max])
# # xlines = [2 * i - 1 for i in range(1, 6)]
# xlines = [ i  for i in range(1, 11)]

# exact_time=expectations['evotime']
# deviation=[]
# exact_J=J_field
# ranks=[]
# evotimes=[]
# for rank in [2,4,8,16,32,64,128,256]:
#     outfile = './Data/Approx/expectations:{}sites-{}up-{}down-{}t0-{}U-{}t_max-{}steps-{}gamma-{}mu-{}rank-{}pbc.npz'.format(
#         L,
#         N_up,
#         N_down,
#         t0, U,
#         t_max,
#         n_steps,
#         gamma,
#         mu, rank,
#         pbc)
#     expectations = np.load(outfile)
#     ranks.append(rank)
#     evotimes.append(expectations['evotime'])
#     J_field=expectations['current']
#     deviation.append(delta*np.sum(np.sqrt(((J_field-exact_J))**2)))
#
# for rank in [2,4,8,16,32,64,128,256]:
#     outfile = './Data/Approx/expectations:{}sites-{}up-{}down-{}t0-{}U-{}t_max-{}steps-{}gamma-{}mu-{}rank-{}pbc.npz'.format(
#         L,
#         N_up,
#         N_down,
#         t0, U,
#         t_max,
#         n_steps,
#         gamma,
#         mu, rank,
#         pbc)
#     expectations = np.load(outfile)
#     J_field=expectations['current']
#     plt.subplot(211)
#     plt.grid(True)
#     if rank==2 or rank==256:
#         plt.plot(times, J_field, label='rank={}'.format(rank))
#     else:
#         plt.plot(times, J_field)
#     plt.legend(loc='upper left')
#     plt.subplot(212)
#     plt.grid(True)
#     exact = np.gradient(J_field, delta)
#     w, spec = spectrum_welch(exact.real, delta)
#     w *= 2. * np.pi
#     plt.semilogy(w, spec)
# for xc in xlines:
#     plt.axvline(x=xc, color='black', linestyle='dashed')
# plt.xlabel('$\\omega$')
# plt.ylabel('Power Spectra')
# # plt.legend(loc='upper right')
# plt.savefig('./Plots/currentsvsrank' + figparams,
#             bbox_inches='tight')
# plt.show()
#
# plt.subplot(211)
# plt.plot(ranks,evotimes,'-x')
# plt.plot(ranks,exact_time*np.ones(len(ranks)),linestyle='--',color='black',label='exact runtime')
# plt.legend()
# plt.ylabel('Runtime(s)')
# plt.subplot(212)
# plt.plot(ranks,deviation,'-x')
# plt.ylabel('$\epsilon$')
# plt.xlabel('Rank')
# plt.savefig('./Plots/runtimeanderror' + figparams,bbox_inches='tight')
# plt.show()

gammas=[]
final_J=[]
# mu=0.9
for gamma_exp in [1,2]:
    # gamma_size=-gamma_exp
    gamma=10**(-gamma_exp)*t0
    gammas.append(10**-gamma_exp)
    outfile = './Data/Approx/expectations:{}sites-{}up-{}down-{}t0-{}U-{}t_max-{}steps-{}gamma-{}mu-{}rank-{}pbc.npz'.format(
        L,
        N_up,
        N_down,
        t0, U,
        t_max,
        n_steps,
        gamma,
        mu, rank,
        pbc)
    expectations = np.load(outfile)
    J_field=2*expectations['current']
    final_J.append(J_field[-1].real)
    plt.subplot(211)
    plt.grid(True)
    if rank==2 or rank==256:
        plt.plot(times, J_field, label='$\\Gamma={}'.format(rank))
    else:
        plt.plot(times, J_field)
    plt.legend(loc='upper left')
    plt.subplot(212)
    plt.grid(True)
    exact = np.gradient(J_field, delta)
    w, spec = spectrum_welch(exact.real, delta)
    w *= 2. * np.pi
    max_value = max(spec)
    max_index = np.where(spec == max_value)
    w_scale = w[max_index]
    print(w_scale)
    w = w / w_scale
    plt.semilogy(w, spec)
for xc in xlines:
    plt.axvline(x=xc, color='black', linestyle='dashed')
plt.xlabel('$\\omega$')
plt.ylabel('Power Spectra')
# plt.legend(loc='upper right')
plt.savefig('./Plots/currentsvsgamma' + figparams,
            bbox_inches='tight')
plt.show()


# mu=0.9
rank=128
plt.subplot(211)
for mu in [0.4,0.6,0.8,1]:
    gammas = []
    final_J = []
    for gamma_exp in [0,1,2,3]:
        # gamma_size=-gamma_exp
        gamma=10**(-gamma_exp)*t0
        gammas.append(10**-gamma_exp)
        outfile = './Data/Approx/expectations:{}sites-{}up-{}down-{}t0-{}U-{}t_max-{}steps-{}gamma-{}mu-{}rank-{}pbc.npz'.format(
            L,
            N_up,
            N_down,
            t0, U,
            t_max,
            n_steps,
            gamma,
            mu, rank,
            pbc)
        expectations = np.load(outfile)
        J_field=2*expectations['current']
        final_J.append(J_field[-1].real)

    # gammas=np.sqrt(gammas)
    plt.loglog(gammas,np.abs(final_J),marker='o',label='$\\mu=$%.1f' % (mu))
    # plt.loglog([gammas[0],gammas[-1]],[np.abs(final_J)[0],np.abs(final_J)[-1]],marker='x')
plt.legend()

# plt.xlabel('$\\Gamma/t_0$')
plt.ylabel('$J_f$')
plt.text(0.1, 1e-4, 'Rank=128',fontsize=35)

rank=16
plt.subplot(212)
for mu in [0.4,0.6,0.8,1]:
    gammas = []
    final_J = []
    for gamma_exp in [0,1,2,3]:
        # gamma_size=-gamma_exp
        gamma=10**(-gamma_exp)*t0
        gammas.append(10**-gamma_exp)
        outfile = './Data/Approx/expectations:{}sites-{}up-{}down-{}t0-{}U-{}t_max-{}steps-{}gamma-{}mu-{}rank-{}pbc.npz'.format(
            L,
            N_up,
            N_down,
            t0, U,
            t_max,
            n_steps,
            gamma,
            mu, rank,
            pbc)
        expectations = np.load(outfile)
        J_field=2*expectations['current']
        final_J.append(J_field[-1].real)
    # gammas=np.sqrt(gammas)
    plt.loglog(gammas,np.abs(final_J),marker='o',label='$\\mu=$%.1f' % (mu))
    # plt.loglog([gammas[0],gammas[-1]],[np.abs(final_J)[0],np.abs(final_J)[-1]],marker='x')
# plt.legend()

plt.xlabel('$\\Gamma/t_0$')
plt.ylabel('$J_f$')

# plt.show()
plt.text(0.1, 1e-4, 'Rank=16',fontsize=35)
plt.show()

rank=128
plt.subplot(211)
for gamma_exp in [0, 1, 2, 3]:
    gammas = []
    final_J = []
    for mu in [0.4,0.6,0.8,1]:
        # gamma_size=-gamma_exp
        gamma=10**(-gamma_exp)*t0
        gammas.append(mu)
        outfile = './Data/Approx/expectations:{}sites-{}up-{}down-{}t0-{}U-{}t_max-{}steps-{}gamma-{}mu-{}rank-{}pbc.npz'.format(
            L,
            N_up,
            N_down,
            t0, U,
            t_max,
            n_steps,
            gamma,
            mu, rank,
            pbc)
        expectations = np.load(outfile)
        J_field=2*expectations['current']
        final_J.append(J_field[-1].real)
    # gammas=np.sqrt(gammas)
    plt.loglog(gammas,np.abs(final_J),marker='o',label='$\\Gamma=10^{-%.1f}$' % (gamma))
    # plt.loglog([gammas[0],gammas[-1]],[np.abs(final_J)[0],np.abs(final_J)[-1]],marker='x')
plt.legend()

# plt.xlabel('$\\Gamma/t_0$')
plt.ylabel('$J_f$')

rank=64
plt.subplot(212)
for gamma_exp in [0, 1, 2, 3]:
    gammas = []
    final_J = []
    for mu in [0.4,0.6,0.8,1]:
        # gamma_size=-gamma_exp
        gamma=10**(-gamma_exp)*t0
        gammas.append(mu)
        outfile = './Data/Approx/expectations:{}sites-{}up-{}down-{}t0-{}U-{}t_max-{}steps-{}gamma-{}mu-{}rank-{}pbc.npz'.format(
            L,
            N_up,
            N_down,
            t0, U,
            t_max,
            n_steps,
            gamma,
            mu, rank,
            pbc)
        expectations = np.load(outfile)
        J_field=2*expectations['current']
        final_J.append(J_field[-1].real)
    # gammas=np.sqrt(gammas)
    plt.loglog(gammas,np.abs(final_J),marker='o',label='$\\Gamma=10^{-%.1f}$' % (gamma))
    # plt.loglog([gammas[0],gammas[-1]],[np.abs(final_J)[0],np.abs(final_J)[-1]],marker='x')
# plt.legend()

plt.xlabel('$\\Gamma/t_0$')
plt.ylabel('$J_f$')

plt.show()
gamma=0.1*t0
rank=128
for mu in [1,0.8,0.6,0.4]:
    # gamma_size=-gamma_exp
    # gamma=10**(-gamma_exp)*np.sqrt(t0)
    outfile = './Data/Approx/expectations:{}sites-{}up-{}down-{}t0-{}U-{}t_max-{}steps-{}gamma-{}mu-{}rank-{}pbc.npz'.format(
        L,
        N_up,
        N_down,
        t0, U,
        t_max,
        n_steps,
        gamma,
        mu, rank,
        pbc)
    expectations = np.load(outfile)
    J_field=2*expectations['current']
    plt.subplot(211)
    plt.xlabel('Time',fontsize=32)
    plt.ylabel('$a(t)$')
    plt.grid(True)
    # if rank==2 or rank==256:
    #     plt.plot(times, J_field, label='$\\Gamma={}'.format(rank))
    # else:
    #     plt.plot(times, J_field)
    exact = np.gradient(J_field, delta)
    # plt.plot(times, J_field, label='$\\mu=${}'.format(mu))
    plt.plot(times, exact, label='$\\mu=${}'.format(mu))
    plt.subplot(212)
    plt.grid(True)
    exact = np.gradient(J_field, delta)
    w, spec = spectrum_welch(exact.real, delta)
    w *= 2. * np.pi
    max_value = max(spec)
    max_index = np.where(spec==max_value)
    w_scale=w[max_index]
    print(w_scale)
    w=w/w_scale
    plt.semilogy(w, spec,label='$\\mu=${}'.format(mu))
    plt.legend(loc='upper right')
    plt.xlim(0,30*w_scale)
for xc in xlines:
    plt.axvline(x=xc, color='black', linestyle='dashed')
plt.xlabel('$\\omega/\\omega_{0}$')
plt.ylabel('$S(\\omega)$')
# plt.legend(loc='upper right')
plt.savefig('./Plots/currentsvsgamma' + figparams,
            bbox_inches='tight')
plt.show()

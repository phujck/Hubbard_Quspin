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
    'axes.labelsize': 30,
    # 'legend.fontsize': 28,
    'legend.fontsize': 23,
    'xtick.labelsize': 22,
    'ytick.labelsize': 22,
    'figure.figsize': [5.2 * 3.375, 3.5 * 3.375],
    'text.usetex': True
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

"""load expectations. For densities, time is first dimension, site number is second"""
J_field = expectations['current']
up_densities = np.vstack([expectations["nup" + str(j)] for j in range(L)]).T
down_densities = np.vstack([expectations["ndown" + str(j)] for j in range(L)]).T
D_densities = np.vstack([expectations["D" + str(j)] for j in range(L)]).T

J_field2 = expectations2['current']
up_densities2 = np.vstack([expectations2["nup" + str(j)] for j in range(L2)]).T
down_densities2 = np.vstack([expectations2["ndown" + str(j)] for j in range(L2)]).T
D_densities2 = np.vstack([expectations2["D" + str(j)] for j in range(L2)]).T

print(up_densities.shape)
print(D_densities2.shape)
"""example for using the analysis code"""
# J_field=expectations['current']
# plt.xlabel("Time (cycles)")
# plt.ylabel("$J(t)$")
# plt.grid(True)
# plt.tight_layout()
# plt.plot(times,J_field)
# plt.savefig('./Plots/current-'+figparams, bbox_inches='tight')
# plt.show()
# plt.close()

"""plotting spin densities"""

"""shows groundstate spin densities for two systems"""
# plt.subplot(211)
# plt.plot(sites, D_densities[0, :], '-x', label='$\\langle D_j\\rangle$')
# plt.plot(sites, up_densities[0, :], '--x', label='$\\langle n_{\\uparrow j} \\rangle$')
# plt.plot(sites, down_densities[0, :], '-.x', label='$\\langle n_{\\downarrow j} \\rangle$')
# plt.title('Groundstate')
#
# plt.subplot(212)
# plt.plot(sites2, D_densities2[0, :], '-x', label='$\\langle D_j\\rangle$')
# plt.plot(sites2, up_densities2[0, :], '--x', label='$\\langle n_{\\uparrow j} \\rangle$')
# plt.plot(sites2, down_densities2[0, :], '-.x', label='$\\langle n_{\\downarrow j} \\rangle$')
# # plt.plot(sites, D_densities.flatten()-up_densities.flatten()*down_densities.flatten(), label='$\\langle D_j\\rangle-\\langle n_\\uparrow j \\rangle \\langle n_\\downarrow j \\rangle$')
# plt.xlabel('site')
# plt.legend()
# plt.savefig('./Plots/groundstate-' + figparams, bbox_inches='tight')
# plt.show()
#
# # """Doublon density dynamics"""
# plt.subplot(211)
# sitelabels = ['$D_%s(t)$' % (j) for j in range(L)]
# print(sitelabels)
# # plt.ylabel('$D_j(t), \\frac{U_b}{t_0}= %.1f, \\frac{SO}{t_0}= %.1f $' % (U_b/t0,SO/t0))
# for i in range(L):
#     plt.plot(times, D_densities[:, i], label=sitelabels[i])
# plt.legend()
# plt.title('Doublon Densities')
# plt.subplot(212)
# plt.plot(times2, D_densities2)
# plt.ylabel('$D_j(t), \\frac{U_b}{t_0}= %.1f$' % (U2 / t02))
# plt.xlabel('Time [cycles]')
# plt.savefig('./Plots/DoublonDensities-' + figparams, bbox_inches='tight')
# plt.show()
#
# plt.subplot(211)
# sitelabels = ['$n_{ \\uparrow %s}(t)$' % (j) for j in range(L)]
# print(sitelabels)
# plt.title('Down densities')
# plt.ylabel('$n_{ \\uparrow j} (t)$')
# for i in range(L):
#     plt.plot(times, up_densities[:, i], label=sitelabels[i])
# plt.legend()
#
# plt.subplot(212)
# plt.plot(times2, up_densities2)
# plt.ylabel('$n_{ \\downarrow j} (t)$')
# plt.xlabel('Time [cycles]')
# plt.savefig('./Plots/spindensities-' + figparams, bbox_inches='tight')
# plt.show()

# """shows up and down densities on the same plot, for one set of parameters"""
# plt.subplot(211)
# sitelabels=['$n_{ \\uparrow %s}(t)$' % (j) for j in range(L)]
# print(sitelabels)
# plt.title('up vs down densities')
# plt.ylabel('$n_{ \\uparrow j} (t), \\frac{U_b}{t_0}= %.1f, \\frac{SO}{t_0}= %.1f $' % (U_b/t0,SO/t0))
# for i in range(L):
#     plt.plot(times,up_densities[:,i],label=sitelabels[i])
# plt.legend()
#
# plt.subplot(212)
# plt.plot(times,down_densities)
# plt.ylabel('$n_{ \\downarrow j} (t), \\frac{U_b}{t_0}= %.1f, \\frac{SO}{t_0}= %.1f $' % (U_b/t0,SO/t0))
# plt.xlabel('Time [cycles]')
# plt.savefig('./Plots/UPvsDownDensities-'+figparams, bbox_inches='tight')
# plt.show()
#
# """shows the excess correlations D-n_up*n_down"""
# plt.subplot(211)
# sitelabels=['$D_%s(t)-n_{ \\uparrow %s}n_{ \\downarrow %s}(t)$' % (j,j,j) for j in range(L)]
# print(sitelabels)
# plt.title('excess doublon correlations,$D_j - n_{ \\downarrow j}n_{ \\uparrow j}$')
# plt.ylabel('$\\frac{U_b}{t_0}= %.1f, \\frac{SO}{t_0}= %.1f $' % (U_b/t0,SO/t0))
# for i in range(L):
#     plt.plot(times,D_densities[:,i]-(up_densities*down_densities)[:,i],label=sitelabels[i])
# plt.legend()
#
# plt.subplot(212)
# plt.plot(times,D_densities2-(up_densities2*down_densities2))
# plt.ylabel('$ \\frac{U_b}{t_0}= %.1f , \\frac{SO}{t_0}= %.1f $' % (U_b2/t02,SO2/t02))
# plt.xlabel('Time [cycles]')
# plt.savefig('./Plots/spincorrelationdifference'+figparams,bbox_inches='tight')
# plt.show()
#
# """shows up-down densities"""
# plt.subplot(211)
# sitelabels=['$n_{ \\uparrow %s}-n_{ \\downarrow %s}(t)$' % (j,j) for j in range(L)]
# print(sitelabels)
# plt.title('up spin - down spin')
# plt.ylabel('$n_{ \\downarrow j}-n_{ \\uparrow j}, \\frac{U_b}{t_0}= %.1f $' % (U/t0))
# for i in range(L):
#     plt.plot(times,up_densities[:,i]-down_densities[:,i],label=sitelabels[i])
# plt.legend()
#
# plt.subplot(212)
# plt.plot(times,up_densities2-down_densities2)
# plt.ylabel('$n_{ \\downarrow j}-n_{ \\uparrow j}, \\frac{U_b}{t_0}= %.1f$' % (U2/t02))
# plt.xlabel('Time [cycles]')
# plt.savefig('./Plots/spindiffs'+figparams,bbox_inches='tight')
# plt.show()


# """plot currents"""
# plt.subplot(211)
# plt.plot(times, J_field)
# # plt.xlabel('Time [cycles]')
# plt.ylabel('$J(t)$')
# # plt.legend(loc='upper right')
# # plt.annotate('a)', xy=(0.3, np.max(J_field) - 0.08), fontsize=25)
# plt.legend(loc=1)
#
# plt.subplot(212)
# plt.plot(times2, J_field2)
# plt.ylabel('$J(t)$')
# # plt.annotate('b)', xy=(0.3, np.max(J_field2) - 0.05), fontsize=25)
# plt.legend(loc=1)
# plt.show()
#
# nn = expectations['neighbour']
# nn2 = expectations2['neighbour']
#
# plt.subplot(211)
# plt.plot(times, nn.real,label='real')
# plt.plot(times, nn.imag,label='imag')
# # plt.xlabel('Time [cycles]')
# plt.ylabel('$\\sum_j \\langle \\hat{c}^\\dagger_j \\hat{c}_j\\rangle (t)$')
# plt.legend(loc='upper right')
# # plt.annotate('a)', xy=(0.3, np.max(J_field) - 0.08), fontsize=25)
# plt.legend(loc=1)
#
# plt.subplot(212)
# plt.plot(times2, nn2.real,label='real')
# plt.plot(times2, nn2.imag,label='imag')
# # plt.xlabel('Time [cycles]')
# plt.ylabel('$\\sum_j \\langle \\hat{c}^\\dagger_j \\hat{c}_j\\rangle (t)$')
# plt.legend(loc='upper right')
# # plt.annotate('a)', xy=(0.3, np.max(J_field) - 0.08), fontsize=25)
# plt.legend(loc=1)
# plt.show()
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
min_spec = 18
max_harm = 15
gabor = 'fL'
exact = np.gradient(J_field, delta)
exact2 = np.gradient(J_field2, delta2)
w, spec = spectrum_welch(exact.real, delta)
w2, spec2 = spectrum_welch(exact2.real, delta2)
w *= 2. * np.pi
w2 *= 2. * np.pi
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

plt.subplot(211)
plt.plot(times, J_field,linestyle='--',label='exact',color='black')
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
w, spec = spectrum_welch(exact.real, delta)
w *= 2. * np.pi
plt.semilogy(w, spec, linestyle='--',color='black')
axes = plt.gca()
axes.set_xlim([0, max_harm])
if spec.max() > prev_max:
    prev_max = spec.max() * 5
axes.set_ylim([10 ** (-min_spec), prev_max])
# xlines = [2 * i - 1 for i in range(1, 6)]
xlines = [ i  for i in range(1, 11)]

exact_time=expectations['evotime']
deviation=[]
exact_J=J_field
ranks=[]
evotimes=[]
for rank in [2,8,16,32,65,127,256]:
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
    ranks.append(rank)
    evotimes.append(expectations['evotime'])
    J_field=expectations['current']
    deviation.append(delta*np.sum(np.sqrt(((J_field-exact_J))**2)))

for rank in [2, 8, 16, 64,128,256]:
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
    J_field=expectations['current']
    plt.subplot(211)
    plt.plot(times, J_field,label='rank={}'.format(rank))
    plt.legend(loc='upper left')
    plt.subplot(212)
    exact = np.gradient(J_field, delta)
    w, spec = spectrum_welch(exact.real, delta)
    w *= 2. * np.pi
    plt.semilogy(w, spec)
for xc in xlines:
    plt.axvline(x=xc, color='black', linestyle='dashed')
plt.xlabel('$\\omega$')
plt.ylabel('HHG spectra')
# plt.legend(loc='upper right')
plt.savefig('./Plots/currentsvsrank' + figparams,
            bbox_inches='tight')
plt.show()

plt.subplot(211)
plt.plot(ranks,evotimes,'-x')
plt.plot(ranks,exact_time*np.ones(len(ranks)),linestyle='--',color='black',label='exact runtime')
plt.legend()
plt.ylabel('Runtime(s)')
plt.subplot(212)
plt.plot(ranks,deviation,'-x')
plt.ylabel('$\epsilon$')
plt.xlabel('Rank')
plt.savefig('./Plots/runtimeanderror' + figparams,bbox_inches='tight')
plt.show()
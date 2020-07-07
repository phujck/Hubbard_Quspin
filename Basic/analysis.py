##########################################################
# Basic Analysis code for the simple Hubbard model       #
##########################################################
import sys
import numpy as np
import matplotlib.pyplot as plt
import numpy.ma as ma
from matplotlib import cm as cm
from scipy.signal import blackman
from scipy.signal import stft
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from scipy import signal

sys.path.append('../')
from tools import parameter_instantiate as hhg  # Used for scaling units.

"""Some helpful stuff for plotting"""

def spectrum_welch(at, delta):
    return signal.welch(at, 1. / delta, nperseg=len(at), scaling='spectrum')



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
L = 10 # system size
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
lat = hhg(field=field, nup=N_up, ndown=N_down, nx=L, ny=0, U=U, t=t0, F0=F0, a=a, pbc=pbc)

"""System Evolution Time"""
cycles = 10  # time in cycles of field frequency
n_steps = 2000
start = 0
# real time
# stop = cycles / lat.freq
# scaling time to frequency
stop = cycles
times, delta = np.linspace(start, stop, num=n_steps, endpoint=True, retstep=True)

"""loading expectations"""
outfile = './Data/expectations:{}sites-{}up-{}down-{}t0-{}U-{}cycles-{}steps-{}pbc.npz'.format(L, N_up, N_down, t0, U,
                                                                                               cycles,
                                                                                               n_steps, pbc)
U2=1.0001*t0
outfile2 = './Data/expectations:{}sites-{}up-{}down-{}t0-{}U-{}cycles-{}steps-{}pbc.npz'.format(L, N_up, N_down, t0, U2,
                                                                                               cycles,
                                                                                               n_steps, pbc)
figparams = '{}sites-{}up-{}down-{}t0-{}U-{}cycles-{}steps-{}pbc.pdf'.format(L, N_up, N_down, t0, U, cycles,
                                                                             n_steps, pbc)
# npzfile = np.load(outfile)
expectations = np.load(outfile)
expectations2 = np.load(outfile2)

"""show the expectations available here"""
print('expectations available: {}'.format(expectations.files))
sites = [j for j in range(L)]
sites2 = sites
"""example for using the analysis code"""
J_field = expectations['current']
J_field2 = expectations2['current']
E=expectations['H']
E2=expectations2['H']

up_densities = np.vstack([expectations["nup" + str(j)] for j in range(L)]).T
down_densities = np.vstack([expectations["ndown" + str(j)] for j in range(L)]).T
D_densities = np.vstack([expectations["D" + str(j)] for j in range(L)]).T

# J_field2 = expectations2['current']
up_densities2 = np.vstack([expectations2["nup" + str(j)] for j in range(L)]).T
down_densities2 = np.vstack([expectations2["ndown" + str(j)] for j in range(L)]).T
D_densities2 = np.vstack([expectations2["D" + str(j)] for j in range(L)]).T

plt.subplot(211)
plt.plot(sites, D_densities[0, :], '-x', label='$\\langle D_j\\rangle$')
plt.plot(sites, up_densities[0, :], '--x', label='$\\langle n_{\\uparrow j} \\rangle$')
plt.plot(sites, down_densities[0, :], '-.x', label='$\\langle n_{\\downarrow j} \\rangle$')
plt.title('Groundstate')

plt.subplot(212)
plt.plot(sites2, D_densities2[0, :], '-x', label='$\\langle D_j\\rangle$')
plt.plot(sites2, up_densities2[0, :], '--x', label='$\\langle n_{\\uparrow j} \\rangle$')
plt.plot(sites2, down_densities2[0, :], '-.x', label='$\\langle n_{\\downarrow j} \\rangle$')
# plt.plot(sites, D_densities.flatten()-up_densities.flatten()*down_densities.flatten(), label='$\\langle D_j\\rangle-\\langle n_\\uparrow j \\rangle \\langle n_\\downarrow j \\rangle$')
plt.xlabel('site')
plt.legend()
plt.savefig('./Plots/groundstate-' + figparams, bbox_inches='tight')
plt.show()

"""Doublon density dynamics"""
plt.subplot(211)
sitelabels=['$D_%s(t)$' % (j) for j in range(L)]
print(sitelabels)
plt.ylabel('$D_j(t), \\frac{U_b}{t_0}= %.1f$' % (U/t0))
for i in range(L):
    plt.plot(times,D_densities[:,i],label=sitelabels[i])
plt.legend()
plt.title('Doublon Densities')
plt.subplot(212)
plt.plot(times,D_densities2)
plt.ylabel('$D_j(t), \\frac{U_b}{t_0}= %.1f$' % (U2/t0,))
plt.xlabel('Time [cycles]')
plt.savefig('./Plots/DoublonDensities-'+figparams, bbox_inches='tight')
plt.show()
#

plt.xlabel("Time (cycles)")
plt.ylabel("$J(t)$")
plt.grid(True)
plt.tight_layout()
plt.plot(times, J_field,label='original')
plt.plot(times, J_field2,label='in symmetry block',linestyle='--')
plt.savefig('./Plots/current-' + figparams, bbox_inches='tight')
plt.legend()
plt.show()

plt.xlabel("Time (cycles)")
plt.ylabel("$H(t)$")
plt.grid(True)
plt.tight_layout()
plt.plot(times, E,label='original')
plt.plot(times, E2,label='in symmetry block',linestyle='--')
plt.savefig('./Plots/energies-' + figparams, bbox_inches='tight')
plt.legend()
plt.show()

prev_max=0
for K in [U,U2]:

    outfile = './Data/expectations:{}sites-{}up-{}down-{}t0-{}U-{}cycles-{}steps-{}pbc.npz'.format(L, N_up, N_down, t0,
                                                                                                   K,
                                                                                                   cycles,
                                                                                                   n_steps, pbc)

    expectations = np.load(outfile)
    J_field = expectations['current'].real
    plt.subplot(211)
    plt.plot(times, J_field, label='$\\frac{U_b}{t_0}= %.1f $' % (K / t0))
    # plt.xlabel('Time [cycles]')
    plt.ylabel('$J(t)$')
    plt.xlabel('Time [cycles]')
    plt.legend(loc='upper right')

    plt.subplot(212)
    method = 'welch'
    min_spec = 20
    max_harm = 60
    gabor = 'fL'
    exact = np.gradient(J_field, delta)
    w, spec = spectrum_welch(exact, delta)
    # w *= 2. * np.pi / lat.field
    plt.semilogy(w, spec, label='$J(t)$')
    axes = plt.gca()
    axes.set_xlim([0, max_harm])
    if spec.max() > prev_max:
        prev_max = spec.max() * 5
    axes.set_ylim([10 ** (-min_spec), prev_max])
    xlines = [2 * i - 1 for i in range(1, 6)]

for xc in xlines:
    plt.axvline(x=xc, color='black', linestyle='dashed')
plt.xlabel('Harmonic Order')
plt.ylabel('HHG spectra')
# plt.legend(loc='upper right')
plt.savefig('./Plots/spinorbitspectracombined' + figparams,
            bbox_inches='tight')
plt.show()



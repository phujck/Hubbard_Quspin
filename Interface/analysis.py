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

"""Some helpful stuff for plotting"""


class hhg:
    def __init__(self, field, nup, ndown, nx, ny, U, t=0.52, SO=0, F0=10., a=4., lat_type='square', pbc=True):
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
        self.SO = SO / t
        if len(self.U) == 1:
            print("U= %.3f t_0" % self.U)
        else:
            print('onsite potential U list:')
            print(self.U)
        print("SO= %.3f t_0" % self.SO)
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
L = 10  # system size
N_up = 5  # number of fermions with spin up
N_down = 5  # number of fermions with spin down
N = N_up + N_down  # number of particles
t0 = 0.52  # hopping strength
SO = 0 * t0  # spin orbit coupling
# U = 0*t0  # interaction strength
U = 1 * t0  # interaction strength
pbc = True

"""Set up the partition of the system's onsite potentials"""
U_a = 0 * t0
U_b = 0 * t0
partition = 5
U = []
for n in range(L):
    # if n < int(nx/2):
    if n < partition:
        U.append(U_a)
    else:
        U.append(U_b)
U = np.array(U)

"""Laser pulse parameters"""
field = 32.9  # field angular frequency THz
F0 = 10  # Field amplitude MV/cm
a = 4  # Lattice constant Angstroms

"""Second set of parameters to mod"""
"""Hubbard model Parameters"""
L2 = 8  # system size
N_up2 = 4  # number of fermions with spin up
N_down2 = 4  # number of fermions with spin down
N2 = N_up2 + N_down2  # number of particles
t02 = 0.52  # hopping strength
SO2 = 10 * t0  # spin orbit coupling
# U = 0*t0  # interaction strength
U2 = 1 * t0  # interaction strength
pbc2 = True

"""Set up the partition of the system's onsite potentials"""
U_a2 = 0 * t0
U_b2 = 0 * t0
partition2 = 4
U2 = []
for n in range(L2):
    # if n < int(nx/2):
    if n < partition2:
        U2.append(U_a2)
    else:
        U2.append(U_b2)
U2 = np.array(U2)

"""Laser pulse parameters"""
field2 = 32.9  # field angular frequency THz
F02 = 10  # Field amplitude MV/cm
a2 = 4  # Lattice constant Angstroms

"""instantiate parameters with proper unit scaling. In this case everything is scaled to units of t_0"""
lat = hhg(field=field, nup=N_up, ndown=N_down, nx=L, ny=0, U=U, SO=SO, t=t0, F0=F0, a=a, pbc=pbc)
lat2 = hhg(field=field2, nup=N_up2, ndown=N_down, nx=L2, ny=0, U=U, SO=SO2, t=t02, F0=F02, a=a2, pbc=pbc2)

"""This is used for setting up Hamiltonian in Quspin."""
dynamic_args = []

"""System Evolution Time"""
cycles = 10  # time in cycles of field frequency
n_steps = 2000
start = 0
stop = cycles / lat.freq
# stop = 0.5
times, delta = np.linspace(start, stop, num=n_steps, endpoint=True, retstep=True)
sites = range(L)
sites2 = range(L2)
# print(times)

"""set up parameters for saving expectations later"""
outfile = './Data/expectations:{}sites-{}up-{}down-{}t0-{}U-{}SO-{}cycles-{}steps-{}pbc.npz'.format(L, N_up, N_down, t0,
                                                                                                    U, SO, cycles,
                                                                                                    n_steps, pbc)
outfile2 = './Data/expectations:{}sites-{}up-{}down-{}t0-{}U-{}SO-{}cycles-{}steps-{}pbc.npz'.format(L2, N_up2, N_down2,
                                                                                                     t02, U2, SO2,
                                                                                                     cycles,
                                                                                                     n_steps, pbc2)

figparams = '{}sites-{}up-{}down-{}t0-{}U-{}SO-{}cycles-{}steps-{}pbc.pdf'.format(L, N_up, N_down, t0, U, SO, cycles,
                                                                                  n_steps, pbc)
expectations = np.load(outfile)
expectations2 = np.load(outfile2)

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
plt.subplot(211)
plt.plot(sites, D_densities[0, :], '-x', label='$\\langle D_j\\rangle$')
plt.plot(sites, up_densities[0, :], '--x', label='$\\langle n_{\\uparrow j} \\rangle$')
plt.plot(sites, down_densities[0, :], '-.x', label='$\\langle n_{\\downarrow j} \\rangle$')
plt.vlines(partition - 0.5, 0, up_densities[0, :].max(), linestyle='dashed')
plt.ylabel('$\\frac{U_b}{t_0}= %.1f, \\frac{SO}{t_0}= %.1f $' % (U_b / t0, SO / t0))
plt.title('Groundstate')

plt.subplot(212)
plt.plot(sites2, D_densities2[0, :], '-x', label='$\\langle D_j\\rangle$')
plt.plot(sites2, up_densities2[0, :], '--x', label='$\\langle n_{\\uparrow j} \\rangle$')
plt.plot(sites2, down_densities2[0, :], '-.x', label='$\\langle n_{\\downarrow j} \\rangle$')
plt.vlines(partition2 - 0.5, 0, up_densities2[0, :].max(), linestyle='dashed')
plt.ylabel('$\\frac{U_b}{t_0}= %.1f, \\frac{SO}{t_0}= %.1f $' % (U_b2 / t02, SO2 / t02))
# plt.plot(sites, D_densities.flatten()-up_densities.flatten()*down_densities.flatten(), label='$\\langle D_j\\rangle-\\langle n_\\uparrow j \\rangle \\langle n_\\downarrow j \\rangle$')
plt.xlabel('site')
plt.legend()
plt.savefig('./Plots/groundstate-' + figparams, bbox_inches='tight')
plt.show()

# """Doublon density dynamics"""
# plt.subplot(211)
# sitelabels=['$D_%s(t)$' % (j) for j in range(L)]
# print(sitelabels)
# plt.ylabel('$D_j(t), \\frac{U_b}{t_0}= %.1f, \\frac{SO}{t_0}= %.1f $' % (U_b/t0,SO/t0))
# for i in range(L):
#     plt.plot(times,D_densities[:,i],label=sitelabels[i])
# plt.legend()
# plt.title('Doublon Densities')
# plt.subplot(212)
# plt.plot(times,D_densities2)
# plt.ylabel('$D_j(t), \\frac{U_b}{t_0}= %.1f, \\frac{SO}{t_0}= %.1f $' % (U_b2/t02,SO2/t02))
# plt.xlabel('Time [cycles]')
# plt.savefig('./Plots/DoublonDensities-'+figparams, bbox_inches='tight')
# plt.show()
# #
#
# """spin up density dynamics"""
# plt.subplot(211)
# sitelabels=['$n_{ \\uparrow %s}(t)$' % (j) for j in range(L)]
# print(sitelabels)
# plt.title('Up Densities')
# plt.ylabel('$n_{ \\uparrow j} (t), \\frac{U_b}{t_0}= %.1f, \\frac{SO}{t_0}= %.1f $' % (U_b/t0,SO/t0))
# for i in range(L):
#     plt.plot(times,up_densities[:,i],label=sitelabels[i])
# plt.legend()
#
# plt.subplot(212)
# plt.plot(times,up_densities2)
# plt.ylabel('$n_{ \\uparrow j} (t), \\frac{U_b}{t_0}= %.1f, \\frac{SO}{t_0}= %.1f $' % (U_b2/t02,SO2/t02))
# plt.xlabel('Time [cycles]')
# plt.savefig('./Plots/upDensities-'+figparams, bbox_inches='tight')
# plt.show()
#
# """spin down density dynamics"""
# plt.subplot(211)
# sitelabels=['$n_{ \\downarrow %s}(t)$' % (j) for j in range(L)]
# print(sitelabels)
# plt.title('Down densities')
# plt.ylabel('$n_{ \\downarrow j} (t), \\frac{U_b}{t_0}= %.1f, \\frac{SO}{t_0}= %.1f $' % (U_b/t0,SO/t0))
# for i in range(L):
#     plt.plot(times,down_densities[:,i],label=sitelabels[i])
# plt.legend()
#
# plt.subplot(212)
# plt.plot(times,down_densities2)
# plt.ylabel('$n_{ \\downarrow j} (t), \\frac{U_b}{t_0}= %.1f, \\frac{SO}{t_0}= %.1f $' % (U_b2/t02,SO2/t02))
# plt.xlabel('Time [cycles]')
# plt.savefig('./Plots/downDensities-'+figparams, bbox_inches='tight')
# plt.show()
#
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
# plt.ylabel('$n_{ \\downarrow j}-n_{ \\uparrow j}, \\frac{U_b}{t_0}= %.1f, \\frac{SO}{t_0}= %.1f $' % (U_b/t0,SO/t0))
# for i in range(L):
#     plt.plot(times,up_densities[:,i]-down_densities[:,i],label=sitelabels[i])
# plt.legend()
#
# plt.subplot(212)
# plt.plot(times,up_densities2-down_densities2)
# plt.ylabel('$n_{ \\downarrow j}-n_{ \\uparrow j}, \\frac{U_b}{t_0}= %.1f,\\frac{SO}{t_0}= %.1f $' % (U_b2/t02,SO2/t02))
# plt.xlabel('Time [cycles]')
# plt.savefig('./Plots/spindiffs'+figparams,bbox_inches='tight')
# plt.show()
# #

"""plot currents"""
plt.subplot(211)
plt.plot(times, J_field, label='$\\frac{U_b}{t_0}= %.1f, \\frac{SO}{t_0}= %.1f $' % (U_b / t0, SO / t0))
# plt.xlabel('Time [cycles]')
plt.ylabel('$J(t)$')
# plt.legend(loc='upper right')
# plt.annotate('a)', xy=(0.3, np.max(J_field) - 0.08), fontsize=25)
plt.legend(loc=1)

plt.subplot(212)
plt.plot(times, J_field2, label='$\\frac{U_b}{t_0}= %.1f, \\frac{SO}{t_0}= %.1f $' % (U_b2 / t02, SO2 / t02))
plt.ylabel('$J(t)$')
# plt.annotate('b)', xy=(0.3, np.max(J_field2) - 0.05), fontsize=25)
plt.legend(loc=1)
plt.show()

"""Spectral Analysis"""

"""plot individual spectra"""
prev_max = 0
U_list = [0 * t0]
S_list = [0 * t0, 2 * t0, 6 * t0, 10 * t0]
U_list = S_list

"""energies"""
for U_b, SO in itertools.product(U_list, S_list):

    U_a = 0 * t0
    U = []
    for n in range(L):
        if n < partition:
            U.append(U_a)
        else:
            U.append(U_b)
    U = np.array(U)
    outfile = './Data/expectations:{}sites-{}up-{}down-{}t0-{}U-{}SO-{}cycles-{}steps-{}pbc.npz'.format(L, N_up, N_down,
                                                                                                        t0, U, SO,
                                                                                                        cycles,
                                                                                                        n_steps, pbc)

    figparams = '{}sites-{}up-{}down-{}t0-{}U-{}SO-{}cycles-{}steps-{}pbc.pdf'.format(L, N_up, N_down, t0, U, SO,
                                                                                      cycles,
                                                                                      n_steps, pbc)
    expectations = np.load(outfile)
    ham = expectations['H'].real
    plt.plot(times, ham, label='$\\frac{U_b}{t_0}= %.1f, \\frac{SO}{t_0}= %.1f $' % (U_b / t0, SO / t0))
    plt.xlabel('Time [cycles]')
    plt.ylabel('$E(t)$')
    plt.xlabel('Time [cycles]')
    plt.legend(loc='upper right')

plt.show()

# for U_b,SO in itertools.product(U_list,S_list):
#
#     U_a = 0 * t0
#     U = []
#     for n in range(L):
#         if n < partition:
#             U.append(U_a)
#         else:
#             U.append(U_b)
#     U = np.array(U)
#     outfile = './Data/expectations:{}sites-{}up-{}down-{}t0-{}U-{}SO-{}cycles-{}steps-{}pbc.npz'.format(L, N_up, N_down, t0, U, SO,cycles,
#                                                                                   n_steps,pbc)
#
#     figparams = '{}sites-{}up-{}down-{}t0-{}U-{}SO-{}cycles-{}steps-{}pbc.pdf'.format(L, N_up, N_down, t0, U, SO,
#                                                                                       cycles,
#                                                                                       n_steps, pbc)
#
#     expectations = np.load(outfile)
#     J_field=expectations['current'].real
#     plt.subplot(211)
#     plt.plot(times, J_field, label='$\\frac{U_b}{t_0}= %.1f, \\frac{SO}{t_0}= %.1f $' % (U_b/t0,SO/t0))
#     # plt.xlabel('Time [cycles]')
#     plt.ylabel('$J(t)$')
#     plt.xlabel('Time [cycles]')
#     plt.legend(loc='upper right')
#
#     plt.subplot(212)
#     method = 'welch'
#     min_spec = 20
#     max_harm = 60
#     gabor = 'fL'
#     exact = np.gradient(J_field, delta)
#     w, spec = spectrum_welch(exact, delta)
#     w *= 2. * np.pi / lat.field
#     plt.semilogy(w, spec, label='$J(t)$')
#     axes = plt.gca()
#     axes.set_xlim([0, max_harm])
#     if spec.max() > prev_max:
#         prev_max=spec.max()*5
#     axes.set_ylim([10 ** (-min_spec), prev_max])
#     xlines = [2 * i - 1 for i in range(1, 6)]
#
#     for xc in xlines:
#         plt.axvline(x=xc, color='black', linestyle='dashed')
#     plt.xlabel('Harmonic Order')
#     plt.ylabel('HHG spectra')
#     # plt.legend(loc='upper right')
#     plt.savefig('./Plots/spinorbitspectra' + figparams,
#             bbox_inches='tight')
#     plt.show()

"""Same as before, but presents everything in a combined plot"""
prev_max = 0
# U_list=[0*t0]
# S_list=[0*t0,2*t0,6*t0,10*t0]
for U_b, SO in itertools.product(U_list, S_list):

    U_a = 0 * t0
    U = []
    for n in range(L):
        if n < partition:
            U.append(U_a)
        else:
            U.append(U_b)
    U = np.array(U)
    outfile = './Data/expectations:{}sites-{}up-{}down-{}t0-{}U-{}SO-{}cycles-{}steps-{}pbc.npz'.format(L, N_up, N_down,
                                                                                                        t0, U, SO,
                                                                                                        cycles,
                                                                                                        n_steps, pbc)

    figparams = '{}sites-{}up-{}down-{}t0-{}U-{}SO-{}cycles-{}steps-{}pbc.pdf'.format(L, N_up, N_down, t0, U, SO,
                                                                                      cycles,
                                                                                      n_steps, pbc)

    expectations = np.load(outfile)
    J_field = expectations['current'].real
    plt.subplot(211)
    plt.plot(times, J_field, label='$\\frac{U_b}{t_0}= %.1f, \\frac{SO}{t_0}= %.1f $' % (U_b / t0, SO / t0))
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
    w *= 2. * np.pi / lat.field
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

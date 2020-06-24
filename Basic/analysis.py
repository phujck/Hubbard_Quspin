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

sys.path.append('../')
from tools import parameter_instantiate as hhg  # Used for scaling units.

"""Some helpful stuff for plotting"""


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
L = 12 # system size
N_up = L // 2 + L % 2  # number of fermions with spin up
N_down = L // 2  # number of fermions with spin down
N = N_up + N_down  # number of particles
t0 = 0.52  # hopping strength
# U = 0*t0  # interaction strength
U = 1 * t0  # interaction strength
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
figparams = '{}sites-{}up-{}down-{}t0-{}U-{}cycles-{}steps-{}pbc.pdf'.format(L, N_up, N_down, t0, U, cycles,
                                                                             n_steps, pbc)
npzfile = np.load(outfile)
expectations = np.load(outfile)
"""show the expectations available here"""
print('expectations available: {}'.format(expectations.files))

"""example for using the analysis code"""
J_field = expectations['current']
plt.xlabel("Time (cycles)")
plt.ylabel("$J(t)$")
plt.grid(True)
plt.tight_layout()
plt.plot(times, J_field)
plt.savefig('./Plots/current-' + figparams, bbox_inches='tight')
plt.show()
plt.close()





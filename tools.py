#########################################################################
# Set of tools that should be useful in simulations. Includes parameter #
# scaling, just to make life a little easier on us! In future the       #
# low-rank approximation tools will be here.                            #
#########################################################################

"""reasonably self explanatory. This class scales parameters to units of t0 (a'.u units)"""


class parameter_instantiate:
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
        if type(self.U) is float:
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

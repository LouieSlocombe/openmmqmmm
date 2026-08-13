import math
import os

from scipy import constants as _sc

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))

_HARTREE_J = _sc.physical_constants["hartree-joule relationship"][0]
_BOHR_M = _sc.physical_constants["Bohr radius"][0]

PLANCK_J_S = _sc.h
BOLTZMANN_J_PER_K = _sc.k
LIGHT_SPEED_CM_PER_S = _sc.c * 100
HC_J_CM = PLANCK_J_S * LIGHT_SPEED_CM_PER_S

HARTREE_TO_J = _HARTREE_J
HARTREE_TO_KJ_PER_MOL = _HARTREE_J * _sc.N_A / 1000
HARTREE_TO_KCAL_PER_MOL = HARTREE_TO_KJ_PER_MOL / _sc.calorie
HARTREE_TO_EV = _sc.physical_constants["hartree-electron volt relationship"][0]
HARTREE_TO_WAVENUMBER = _sc.physical_constants["hartree-inverse meter relationship"][0] / 100
KCAL_TO_KJ = _sc.calorie

BOHR_TO_ANG = _BOHR_M * 1e10
ANG_TO_BOHR = 1 / BOHR_TO_ANG
BOHR_TO_NM = _BOHR_M * 1e9
BOHR_TO_M = _BOHR_M
ANG_TO_M = _sc.angstrom

AMU_TO_KG = _sc.u

#: Gradient units: OpenMM works in kJ/mol/nm, the QM codes in Eh/Bohr.
HARTREE_PER_BOHR_TO_KJ_PER_MOL_NM = HARTREE_TO_KJ_PER_MOL / BOHR_TO_NM

GAS_CONSTANT_HARTREE_PER_K = _sc.k / _HARTREE_J
GAS_CONSTANT_KCAL_PER_MOL_K = _sc.R / _sc.calorie / 1000
PLANCK_HARTREE_S = _sc.h / _HARTREE_J
HALF_HC_HARTREE_PER_WAVENUMBER = 0.5 / HARTREE_TO_WAVENUMBER
GHZ_TO_WAVENUMBER = 1e9 / LIGHT_SPEED_CM_PER_S

#: h/(8 pi^2) expressed so that B[GHz] = ROT_CONSTANT_GHZ_AMU_ANG2 / I[amu Ang^2].
ROT_CONSTANT_GHZ_AMU_ANG2 = _sc.h / (8 * math.pi**2 * _sc.u * 1e-20) / 1e9

#: Sackur-Tetrode prefactor for q_trans = PREFACTOR * T^2.5 * M^1.5 / p, with M in amu and p in atm.
TRANS_PARTITION_PREFACTOR = ((2 * math.pi * _sc.u * _sc.k / _sc.h**2) ** 1.5) * _sc.k / _sc.atm

#: N_A pi e^2 / (3 * 4 pi eps_0 * c^2 * u), in km/mol per squared atomic-unit dipole derivative.
IR_INTENSITY_AU_TO_KM_PER_MOL = (
    _sc.N_A * math.pi * _sc.e**2 / (3 * 4 * math.pi * _sc.epsilon_0 * _sc.c**2 * _sc.u) * 1e-3
)

ENERGY_UNIT_FROM_HARTREE = {
    "Eh": 1.0,
    "mEh": 1000.0,
    "eV": HARTREE_TO_EV,
    "meV": HARTREE_TO_EV * 1000,
    "kcal/mol": HARTREE_TO_KCAL_PER_MOL,
    "kcalpermol": HARTREE_TO_KCAL_PER_MOL,
    "kJ/mol": HARTREE_TO_KJ_PER_MOL,
    "kJpermol": HARTREE_TO_KJ_PER_MOL,
    "cm-1": HARTREE_TO_WAVENUMBER,
}

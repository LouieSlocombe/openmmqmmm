import os
from math import isclose

import numpy as np

from ash.interfaces.interface_geometric_new import geomeTRICOptimizer
from ash.interfaces.interface_xtb import xTBTheory
from ash.modules.module_coords import Fragment
from ash.modules.module_singlepoint import Singlepoint


def test_xtb_load():
    coords = """
    O       -1.377626260      0.000000000     -1.740199718
    H       -1.377626260      0.759337000     -1.144156718
    H       -1.377626260     -0.759337000     -1.144156718
    """
    # Defining fragment
    H2O = Fragment(coordsstring=coords, charge=0, mult=1)
    xtb_default = xTBTheory()
    assert xtb_default.xtbmethod == 'GFN1'

    xtb_default.cleanup()
    result_inp = Singlepoint(theory=xtb_default, fragment=H2O)
    energy_inputfile = result_inp.energy
    result_inp2 = Singlepoint(theory=xtb_default, fragment=H2O, Grad=True)
    energy_inputfile = result_inp2.energy
    grad_inputfile = result_inp2.gradient

    # Specifying dummy xtbdir and changing xtbmethod
    xtb_with_dir_gfn2 = xTBTheory(xtbdir='/path/to/xtb',
                                  xtbmethod='GFN2',
                                  runmode='inputfile')
    assert xtb_with_dir_gfn2.xtbdir == '/path/to/xtb'
    assert xtb_with_dir_gfn2.xtbmethod == 'GFN2'

    # Specifying library input
    xtb_library = xTBTheory(runmode='library')

    result_lib = Singlepoint(theory=xtb_library, fragment=H2O)
    energy_library = result_lib.energy

    result_lib2 = Singlepoint(theory=xtb_library, fragment=H2O, Grad=True)
    energy_library = result_lib2.energy
    grad_library = result_lib2.gradient
    refenergy = -5.768502689118895
    assert isclose(energy_library, energy_inputfile)
    assert isclose(energy_library, refenergy)

    # Comparing gradients
    assert np.allclose(grad_inputfile, grad_library, rtol=1e-4)

    xtb_default.cleanup()
    xtb_library.cleanup()
    os.remove('xtb_.engrad')
    os.remove('gradient')
    os.remove('energy')
    os.remove('ASH_SP.result')


def test_geometric_dummy():
    # Define coordinate string
    coords = """
    O       -1.377626260      0.000000000     -1.740199718
    H       -1.377626260      0.759337000     -1.144156718
    H       -1.377626260     -0.759337000     -1.144156718
    """
    # Defining fragment
    H2Ofragment = Fragment(coordsstring=coords, charge=0, mult=1)

    # Defining xTB theory
    zerotheorycalc = xTBTheory()

    # Optimize with xTB theory
    result = geomeTRICOptimizer(fragment=H2Ofragment, theory=zerotheorycalc)

    if result.energy == 0.0:
        print("ASH and geometric finished. Everything looks good")

    os.remove("ASH_Optimizer.result")
    os.remove("Fragment-currentgeo.xyz")
    os.remove("Fragment-optimized.xyz")
    os.remove("Fragment-optimized.ygg")
    os.remove("geometric_OPTtraj.log")
    os.remove("geometric_OPTtraj_optim.xyz")
    os.remove("initialxyzfiletric.xyz")
    os.remove('charges')
    os.remove('energy')
    os.remove('gradient')
    os.remove('xtb_.engrad')
    os.remove('xtb_.out')
    os.remove('xtb_.xyz')

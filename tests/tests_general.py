import os
import shutil

from ash.interfaces.interface_geometric_new import geomeTRICOptimizer
from ash.modules.module_coords import Fragment
from ash.modules.module_singlepoint import ZeroTheory


def test_chargemult():
    # Testing charge/mult definitions

    fragcoords = """
    H 0.0 0.0 0.0
    F 0.0 0.0 1.0
    """
    HF_frag = Fragment(coordsstring=fragcoords)
    HF_frag2 = Fragment(coordsstring=fragcoords, charge=0, mult=1)

    assert HF_frag.charge is None, "Charge is not None"
    assert HF_frag2.charge == 0, "Charge is not 0"


def test_simple_optimization():
    # Define coordinate string
    coords = """
    O       -1.377626260      0.000000000     -1.740199718
    H       -1.377626260      0.759337000     -1.144156718
    H       -1.377626260     -0.759337000     -1.144156718
    """
    # Defining fragment
    H2Ofragment = Fragment(coordsstring=coords, charge=0, mult=1)

    # Defining dummy theory
    zerotheorycalc = ZeroTheory()

    # Optimize with dummy theory
    result = geomeTRICOptimizer(fragment=H2Ofragment, theory=zerotheorycalc)
    # remove directory created by geomeTRIC
    if os.path.exists("geometric_tmpdir"):
        shutil.rmtree("geometric_OPTtraj.tmp")

    os.remove("ASH_Optimizer.result")
    os.remove("Fragment-currentgeo.xyz")
    os.remove("Fragment-optimized.xyz")
    os.remove("Fragment-optimized.ygg")
    os.remove("geometric_OPTtraj.log")
    os.remove("geometric_OPTtraj_optim.xyz")
    os.remove("initialxyzfiletric.xyz")
    shutil.rmtree("geometric_OPTtraj.tmp")

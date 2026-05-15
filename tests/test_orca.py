import os

from ash.interfaces.interface_ORCA import ORCATheory
from ash.modules.module_coords import Fragment
from ash.modules.module_singlepoint import Singlepoint


def test_orca_sp():
    """
    Simple Singlepoint ORCA calculation with charge/mult in fragment
    """
    # ORCA
    orcasimpleinput = "! BP86 def2-SVP tightscf notrah"
    orcablocks = "%scf maxiter 200 end"

    fragcoords = """
    H 0.0 0.0 0.0
    F 0.0 0.0 1.0
    """

    # Add coordinates to fragment
    hf_frag = Fragment(coordsstring=fragcoords,
                       charge=0,
                       mult=1)

    orca_sp_calculation = ORCATheory(orcasimpleinput=orcasimpleinput,
                                     orcablocks=orcablocks)

    result = Singlepoint(theory=orca_sp_calculation,
                         fragment=hf_frag)

    # Clean up
    orca_sp_calculation.cleanup()

    # Reference energy
    ref = -100.350611851152
    threshold = 1e-9
    assert abs(result.energy - ref) < threshold, "Energy-error above threshold"
    os.remove('ASH_SP.result')


def test_orca_grad():
    """
    Simple Singlepoint ORCA calculation with charge/mult in fragment
    """
    # ORCA
    orcasimpleinput = "! BP86 def2-SVP tightscf notrah"
    orcablocks = "%scf maxiter 200 end"

    fragcoords = """
    H 0.0 0.0 0.0
    F 0.0 0.0 1.0
    """

    # Add coordinates to fragment
    hf_frag = Fragment(coordsstring=fragcoords, charge=0, mult=1)

    orca_sp_calculation = ORCATheory(orcasimpleinput=orcasimpleinput, orcablocks=orcablocks)

    result = Singlepoint(theory=orca_sp_calculation, fragment=hf_frag)

    # Clean up
    orca_sp_calculation.cleanup()

    # Reference energy
    ref = -100.350611851152
    threshold = 1e-9
    assert abs(result.energy - ref) < threshold, "Energy-error above threshold"

    # Singlepoint gradient
    result2 = Singlepoint(theory=orca_sp_calculation, fragment=hf_frag, Grad=True)

    orca_sp_calculation.cleanup()

    print("Gradient:", result2.gradient)
    assert abs(result2.energy - ref) < threshold, "Energy-error above threshold"
    os.remove('ASH_SP.result')


def test_orca_sp2():
    """
    Simple Singlepoint ORCA calculation with charge/mult in Singlepoint function
    """
    # ORCA
    orcasimpleinput = "! BP86 def2-SVP tightscf notrah"
    orcablocks = "%scf maxiter 200 end"

    fragcoords = """
    H 0.0 0.0 0.0
    F 0.0 0.0 1.0
    """

    # Add coordinates to fragment
    hf_frag = Fragment(coordsstring=fragcoords)

    orca_sp_calculation = ORCATheory(orcasimpleinput=orcasimpleinput,
                                     orcablocks=orcablocks)

    result = Singlepoint(theory=orca_sp_calculation,
                         fragment=hf_frag,
                         charge=0,
                         mult=1)

    # Clean up
    orca_sp_calculation.cleanup()

    # Reference energy
    ref = -100.350611851152
    threshold = 1e-9
    assert abs(result.energy - ref) < threshold, "Energy-error above threshold"
    os.remove('ASH_SP.result')


def test_orca_bs_sp():
    """
    Singlepoint Broken-symmetry ORCA calculation with charge/mult in fragment
    """
    # ORCA
    orcasimpleinput = "! BP86 def2-SVP  tightscf notrah slowconv"
    orcablocks = """
    %scf
    maxiter 500
    diismaxeq 20
    directresetfreq 1
    end
    """

    fragcoords = """
    Fe 0.0 0.0 0.0
    Fe 0.0 0.0 3.0
    """

    # Add coordinates to fragment
    fe2_frag = Fragment(coordsstring=fragcoords,
                        charge=6,
                        mult=1)
    orca_bs_calc = ORCATheory(orcasimpleinput=orcasimpleinput,
                              orcablocks=orcablocks,
                              brokensym=True,
                              HSmult=11,
                              atomstoflip=[1])
    # Simple Energy SP calc
    result = Singlepoint(fragment=fe2_frag,
                         theory=orca_bs_calc)
    print("energy:", result.energy)
    # Clean up
    orca_bs_calc.cleanup()

    # Reference energy
    ref = -2521.60831655367
    threshold = 1e-6
    assert abs(result.energy - ref) < threshold, "Energy-error above threshold"
    os.remove('ASH_SP.result')

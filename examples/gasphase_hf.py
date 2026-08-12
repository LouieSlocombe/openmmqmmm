"""Gas-phase ORCA calculation: energy, geometry optimization and frequencies.

Run with:
    python examples/gasphase_hf.py

Requires an ORCA installation (see README.md for how it is located).
"""

from openmmqmmm import (
    Fragment,
    ORCATheory,
    configure_logging,
    numerical_frequencies,
    optimize_geometry,
    single_point,
)

# The package is silent by default; this restores the classic console output.
configure_logging()

coords = """
H 0.0 0.0 0.0
F 0.0 0.0 1.0
"""
hf_fragment = Fragment(coordsstring=coords, charge=0, mult=1)

orca_calc = ORCATheory(orcasimpleinput="! r2SCAN def2-SVP def2/J tightscf", orcablocks="%scf maxiter 200 end")

# Each job function returns a Results object and writes it to results_*.json.
single_point_result = single_point(theory=orca_calc, fragment=hf_fragment)
print(f"Single-point energy: {single_point_result.energy:.8f} Eh")

# optimize_geometry updates the fragment's coordinates in place.
optimize_result = optimize_geometry(theory=orca_calc, fragment=hf_fragment)
print(f"Optimized energy:    {optimize_result.energy:.8f} Eh")

freq_result = numerical_frequencies(theory=orca_calc, fragment=hf_fragment)
print(f"Frequencies (cm-1):  {freq_result.frequencies}")

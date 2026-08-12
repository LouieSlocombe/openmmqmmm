"""Checks on the single per-element table and the views derived from it.

The per-element data used to live in eight hand-maintained parallel tables across three
modules. They drifted: two copies of the Alvarez covalent radii disagreed about Na, K and
the M dummy site, and the connectivity code read the copy without the overrides. The
tables are now columns of one row each, so the tests here check the derived views agree
with the table and that the table itself is complete and ordered.
"""

import numpy as np
import pytest

from openmmqmmm import Fragment
from openmmqmmm.coords import atommasses, eldict_covrad, elematomnumbers
from openmmqmmm.elements import (
    _ELEMENT_TABLE,
    atomtypes_dict,
    cm5_dz,
    cm5_radii,
    element_dict_atname,
    element_dict_atnum,
)

# Covered elements: the M dummy site at Z=0, then hydrogen through oganesson.
EXPECTED_ELEMENT_COUNT = 119
# Standard atomic weights are tabulated up to lawrencium.
HIGHEST_Z_WITH_A_MASS = 103


def test_table_is_ordered_and_complete():
    assert len(_ELEMENT_TABLE) == EXPECTED_ELEMENT_COUNT
    assert [row[0] for row in _ELEMENT_TABLE] == list(range(EXPECTED_ELEMENT_COUNT))
    assert _ELEMENT_TABLE[0][1] == "M", "Z=0 is the dummy site"
    assert _ELEMENT_TABLE[1][1] == "H"
    assert _ELEMENT_TABLE[-1][1] == "Og"


def test_symbols_are_unique_and_capitalised():
    symbols = [row[1] for row in _ELEMENT_TABLE]
    assert len(set(symbols)) == len(symbols)
    assert all(symbol == symbol.capitalize() for symbol in symbols)


@pytest.mark.parametrize("row", _ELEMENT_TABLE, ids=lambda row: row[1])
def test_every_view_agrees_with_its_row(row):
    """One row per element is the point: no view may disagree with the row it came from."""
    atomic_number, symbol, name, covalent_radius, mass, cm5_radius, dz = row

    assert elematomnumbers[symbol.lower()] == atomic_number
    assert element_dict_atname[symbol.lower()] == element_dict_atnum[atomic_number]
    assert element_dict_atnum[atomic_number].name == name
    assert element_dict_atnum[atomic_number].symbol == symbol

    if covalent_radius is None:
        assert symbol not in eldict_covrad
    else:
        assert eldict_covrad[symbol] == covalent_radius

    if atomic_number >= 1:
        assert cm5_radii[atomic_number - 1] == cm5_radius
        assert cm5_dz[atomic_number - 1] == dz
    if mass is not None:
        assert atommasses[atomic_number - 1] == mass


def test_mass_table_is_contiguous_from_hydrogen():
    """Atommasses is indexed by Z-1, so a gap would silently shift every later element."""
    assert len(atommasses) == HIGHEST_Z_WITH_A_MASS
    assert all(mass is not None for mass in atommasses)
    assert [row[4] for row in _ELEMENT_TABLE[1 : HIGHEST_Z_WITH_A_MASS + 1]] == atommasses
    # Nothing past lawrencium carries a mass, so the list cannot be extended by accident.
    assert all(row[4] is None for row in _ELEMENT_TABLE[HIGHEST_Z_WITH_A_MASS + 1 :])


def test_cm5_parameters_cover_every_real_element():
    """calc_cm5 indexes these by Z-1 for Z=1..118 with no bounds check of its own."""
    assert len(cm5_radii) == len(cm5_dz) == EXPECTED_ELEMENT_COUNT - 1
    assert not np.isnan(cm5_radii).any()
    assert not np.isnan(cm5_dz).any()


def test_ion_and_dummy_radius_overrides_are_present():
    """These three are deliberately not their Alvarez values, and losing them is silent.

    Alvarez gives Na 1.66 and K 2.03. At those radii a solvated ion is reported as
    covalently bonded to its whole first solvation shell, and every fragment containing
    one becomes a single connected molecule.
    """
    assert eldict_covrad["Na"] == 0.0001
    assert eldict_covrad["K"] == 0.0001
    assert eldict_covrad["M"] == 0.0


def test_atomtypes_map_to_known_elements():
    assert len(atomtypes_dict) == 249
    for atomtype, symbol in atomtypes_dict.items():
        assert symbol.lower() in element_dict_atname, f"atom type {atomtype} maps to unknown element {symbol}"
    # The M-site atom type, which the PDB/GROMACS readers rely on
    assert atomtypes_dict["MW"] == "M"


def test_masses_are_used_end_to_end():
    """A Fragment computes its mass through the table; water is about 18 amu."""
    water = Fragment(
        coords=np.array([[0.0, 0.0, 0.117], [0.0, 0.757, -0.469], [0.0, -0.757, -0.469]]),
        elems=["O", "H", "H"],
        charge=0,
        mult=1,
    )
    assert water.mass == pytest.approx(18.015, abs=0.01)

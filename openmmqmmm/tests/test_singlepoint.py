"""Tests for the single-point job functions.

These are the package's main entry points and were only covered incidentally, at 20%
of statements. ZeroTheory makes every path here runnable with no QM program: what is
checked is the bookkeeping — which energies land in which Results field, how labels
and stoichiometry are handled, and which bad inputs raise.
"""

import numpy as np
import pytest

from openmmqmmm import (
    Fragment,
    Reaction,
    ZeroTheory,
    reaction_energy,
    single_point,
    single_point_fragments,
    single_point_fragments_and_theories,
    single_point_reaction,
    single_point_theories,
)
from openmmqmmm.exceptions import InputError

HF_COORDS = "H 0.0 0.0 0.0\nF 0.0 0.0 0.95\n"


@pytest.fixture
def hydrogen_fluoride():
    return Fragment(coordsstring=HF_COORDS, charge=0, mult=1)


def _labelled_fragments(count=3):
    return [
        Fragment(coordsstring=f"H 0.0 0.0 0.0\nF 0.0 0.0 {0.9 + 0.1 * i}\n", charge=0, mult=1, label=f"frag{i}")
        for i in range(count)
    ]


def test_single_point_energy(hydrogen_fluoride):
    result = single_point(theory=ZeroTheory(), fragment=hydrogen_fluoride)

    assert result.energy == 0.0
    assert result.gradient is None, "No gradient unless one was requested"


def test_single_point_gradient(hydrogen_fluoride):
    result = single_point(theory=ZeroTheory(), fragment=hydrogen_fluoride, grad=True)

    assert result.energy == 0.0
    assert np.shape(result.gradient) == (hydrogen_fluoride.numatoms, 3), "One gradient row per atom"


def test_single_point_requires_a_fragment():
    with pytest.raises(InputError):
        single_point(theory=ZeroTheory(), fragment=None)


def test_single_point_requires_a_theory(hydrogen_fluoride):
    with pytest.raises(InputError):
        single_point(theory=None, fragment=hydrogen_fluoride)


def test_single_point_theories(hydrogen_fluoride):
    """One fragment through several theories gives one energy per theory."""
    theories = [ZeroTheory(), ZeroTheory()]

    result = single_point_theories(theories=theories, fragment=hydrogen_fluoride)

    assert len(result.energies) == len(theories)
    assert all(energy == 0.0 for energy in result.energies)


def test_single_point_theories_with_explicit_charge_and_mult():
    """Charge and multiplicity may come as arguments rather than from the fragment.

    Printing the summary table used to raise TypeError on this path.
    """
    fragment = Fragment(coordsstring=HF_COORDS)

    result = single_point_theories(theories=[ZeroTheory()], fragment=fragment, charge=0, mult=1)

    assert result.energies == [0.0]


def test_single_point_fragments():
    fragments = _labelled_fragments()

    result = single_point_fragments(theory=ZeroTheory(), fragments=fragments)

    assert len(result.energies) == len(fragments)
    assert all(energy == 0.0 for energy in result.energies)


def test_single_point_fragments_and_theories():
    """Every fragment is run through every theory: one row of energies per theory."""
    fragments = _labelled_fragments(2)
    theories = [ZeroTheory(), ZeroTheory()]

    result = single_point_fragments_and_theories(theories=theories, fragments=fragments)

    assert len(result.energies) == len(theories)
    assert all(len(row) == len(fragments) for row in result.energies)


def test_reaction_energy_of_a_null_reaction():
    """With identical energies on both sides the reaction energy is zero."""
    reactant, product = _labelled_fragments(2)
    reaction = Reaction(fragments=[reactant, product], stoichiometry=[-1, 1])

    single_point_reaction(theory=ZeroTheory(), reaction=reaction)

    assert reaction.reaction_energy == pytest.approx(0.0)


def test_reaction_energy_applies_stoichiometry():
    """reaction_energy weights each fragment energy by its coefficient."""
    energy, _unit = reaction_energy(list_of_energies=[-1.0, -2.0], stoichiometry=[-1, 1], unit="Eh", silent=True)

    assert energy == pytest.approx(-1.0)


def test_reaction_rejects_mismatched_stoichiometry():
    fragments = _labelled_fragments(2)

    with pytest.raises(InputError):
        Reaction(fragments=fragments, stoichiometry=[-1])

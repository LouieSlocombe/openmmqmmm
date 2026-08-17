import numpy as np
import pytest

from openmmqmmm import (
    Fragment,
    OpenMMTheory,
    QMMMTheory,
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


def _dummy_qmmm(**kwargs):
    """A cheap real QM/MM object: two well-separated H atoms, QM region is atom 0."""
    fragment = Fragment(elems=["H", "H"], coords=[[1.0, 0, 0], [5.0, 0, 0]], charge=0, mult=1, conncalc=False)
    mm_theory = OpenMMTheory(
        fragment=fragment,
        dummysystem=True,
        platform="Reference",
        autoconstraints=None,
        rigidwater=False,
        hydrogenmass=None,
    )
    qmmm = QMMMTheory(
        fragment=fragment,
        qm_theory=ZeroTheory(),
        mm_theory=mm_theory,
        qmatoms=[0],
        embedding="elstat",
        qm_charge=0,
        qm_mult=1,
        dipole_correction=False,
        **kwargs,
    )
    return qmmm, fragment


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
    theories = [ZeroTheory(), ZeroTheory()]

    result = single_point_theories(theories=theories, fragment=hydrogen_fluoride)

    assert len(result.energies) == len(theories)
    assert all(energy == 0.0 for energy in result.energies)


def test_single_point_theories_with_explicit_charge_and_mult():
    fragment = Fragment(coordsstring=HF_COORDS)

    result = single_point_theories(theories=[ZeroTheory()], fragment=fragment, charge=0, mult=1)

    assert result.energies == [0.0]


def test_qmmm_theory_carries_a_label():
    """The energy tables read theory.label, so every theory class must define one."""
    qmmm, _fragment = _dummy_qmmm()

    assert qmmm.label == "QM/MM"


def test_qmmm_theory_label_can_be_overridden():
    qmmm, _fragment = _dummy_qmmm(label="active-site")

    assert qmmm.label == "active-site"


def test_single_point_theories_completes_with_a_qmmm_theory():
    """The summary table runs after every energy, so a missing label would waste the whole job."""
    qmmm, fragment = _dummy_qmmm()

    result = single_point_theories(theories=[qmmm, ZeroTheory()], fragment=fragment)

    assert result.energies == [0.0, 0.0]


def test_single_point_fragments_and_theories_completes_with_a_qmmm_theory():
    qmmm, fragment = _dummy_qmmm()
    fragment.label = "frag0"

    result = single_point_fragments_and_theories(theories=[qmmm], fragments=[fragment])

    assert result.energies == [[0.0]]


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
    energy, _unit = reaction_energy(list_of_energies=[-1.0, -2.0], stoichiometry=[-1, 1], unit="Eh", silent=True)

    assert energy == pytest.approx(-1.0)


def test_reaction_rejects_mismatched_stoichiometry():
    fragments = _labelled_fragments(2)

    with pytest.raises(InputError):
        Reaction(fragments=fragments, stoichiometry=[-1])

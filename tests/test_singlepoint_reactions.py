"""single_point_fragments, single_point_reaction and reaction_energy branches of openmmqmmm.singlepoint."""

import logging
import re

import pytest
from conftest import labelled_fragments

from openmmqmmm import (
    ORCATheory,
    Reaction,
    ZeroTheory,
    reaction_energy,
    single_point_fragments,
    single_point_fragments_and_theories,
    single_point_reaction,
    single_point_theories,
)
from openmmqmmm.exceptions import InputError

# CODATA 2018 conversion factors, independent of openmmqmmm.constants.
HARTREE_TO_KCAL_PER_MOL = 627.5094740631
HARTREE_TO_EV = 27.211386245988


class _QueuedTheory(ZeroTheory):
    """Returns the queued energies in run order and records the orbital file set before each run."""

    def __init__(self, energies):
        super().__init__()
        self.queue = iter(energies)
        self.moreadfile = None
        self.seen_moreadfiles = []

    def run(self, **_kwargs):
        self.seen_moreadfiles.append(self.moreadfile)
        self.energy = next(self.queue)
        return self.energy


def test_single_point_theories_requires_a_fragment():
    with pytest.raises(InputError, match="requires a fragment"):
        single_point_theories(theories=[ZeroTheory()], fragment=None)


def test_single_point_fragments_requires_a_theory():
    with pytest.raises(InputError, match="requires a theory"):
        single_point_fragments(theory=None, fragments=labelled_fragments(1))


def test_single_point_fragments_requires_charge_and_multiplicity_on_every_fragment():
    fragments = labelled_fragments(3)
    fragments[1].mult = None
    fragments[2].charge = None

    with pytest.raises(InputError, match=r"indices \[1, 2\] are missing charge or multiplicity"):
        single_point_fragments(theory=ZeroTheory(), fragments=fragments)


def test_single_point_fragments_relative_energies_are_measured_from_the_lowest():
    fragments = labelled_fragments(3)
    theory = _QueuedTheory([-1.0, -1.5, -0.5])

    result = single_point_fragments(theory=theory, fragments=fragments, relative_energies=True, unit="kcal/mol")

    assert result.energies == [-1.0, -1.5, -0.5]
    assert result.relative_energies == pytest.approx([0.5 * HARTREE_TO_KCAL_PER_MOL, 0.0, HARTREE_TO_KCAL_PER_MOL])
    assert result.labels == ["frag0", "frag1", "frag2"]
    assert [fragment.energy for fragment in fragments] == [-1.0, -1.5, -0.5]


def test_single_point_fragments_reaction_energy_follows_the_stoichiometry():
    theory = _QueuedTheory([-1.0, -1.5, -0.5])

    result = single_point_fragments(theory=theory, fragments=labelled_fragments(3), stoichiometry=[-1, 1, 0], unit="eV")

    assert result.reaction_energy == pytest.approx(-0.5 * HARTREE_TO_EV)


def test_single_point_fragments_hands_each_fragment_its_orbital_file():
    theory = _QueuedTheory([-1.0, -1.5])

    single_point_fragments(theory=theory, fragments=labelled_fragments(2), moreadfiles=["a.gbw", "b.gbw"])

    assert theory.seen_moreadfiles == ["a.gbw", "b.gbw"]


def test_fragments_and_theories_requires_fragments():
    with pytest.raises(InputError, match="at least one fragment"):
        single_point_fragments_and_theories(theories=[ZeroTheory()], fragments=[])


def test_fragments_and_theories_reports_one_reaction_energy_per_theory():
    theories = [_QueuedTheory([-1.0, -1.5]), _QueuedTheory([-2.0, -2.25])]

    result = single_point_fragments_and_theories(
        theories=theories, fragments=labelled_fragments(2), stoichiometry=[-1, 1]
    )

    assert result.energies == [[-1.0, -1.5], [-2.0, -2.25]]
    assert result.reaction_energies == pytest.approx([-0.5 * HARTREE_TO_KCAL_PER_MOL, -0.25 * HARTREE_TO_KCAL_PER_MOL])


@pytest.mark.parametrize("missing", ["theory", "reaction"])
def test_single_point_reaction_requires_theory_and_reaction(missing):
    kwargs = {"theory": ZeroTheory(), "reaction": Reaction(fragments=labelled_fragments(2), stoichiometry=[-1, 1])}
    kwargs[missing] = None

    with pytest.raises(InputError, match="requires a theory and a reaction"):
        single_point_reaction(**kwargs)


def test_single_point_reaction_rejects_the_wrong_number_of_orbital_files():
    reaction = Reaction(fragments=labelled_fragments(2), stoichiometry=[-1, 1])

    with pytest.raises(InputError, match="1 files for 2 reaction fragments"):
        single_point_reaction(theory=ZeroTheory(), reaction=reaction, moreadfiles=["only.gbw"])


def test_single_point_reaction_uses_the_listed_orbital_files_and_converts_the_energy():
    reaction = Reaction(fragments=labelled_fragments(2), stoichiometry=[-1, 1], unit="eV")
    theory = _QueuedTheory([-1.0, -1.25])

    result = single_point_reaction(theory=theory, reaction=reaction, moreadfiles=["reactant.gbw", "product.gbw"])

    assert theory.seen_moreadfiles == ["reactant.gbw", "product.gbw"]
    assert result.energies == reaction.energies == [-1.0, -1.25]
    assert result.reaction_energy == reaction.reaction_energy == pytest.approx(-0.25 * HARTREE_TO_EV)


class _IceCIORCATheory(ORCATheory):
    """ORCATheory whose run() replays queued ICE-CI energies and the properties ORCATheory.run would parse."""

    def __init__(self, energies, with_properties=True, **options):
        super().__init__(orcasimpleinput="! HF def2-SVP", **options)
        self.queue = iter(energies)
        self.with_properties = with_properties

    def run(self, **_kwargs):
        self.energy = next(self.queue)
        if self.with_properties:
            self.properties = {
                "E_var": self.energy,
                "E_PT2_rest": self.energy / 100,
                "num_genCFGs": 10,
                "num_selected_CFGs": 34,
                "num_after_SD_CFGs": 90,
            }
        return self.energy


@pytest.mark.usefixtures("fake_orca_dir")
def test_single_point_reaction_collects_ice_ci_properties_of_an_orca_theory():
    reaction = Reaction(fragments=labelled_fragments(2), stoichiometry=[-1, 1])
    theory = _IceCIORCATheory([-1.0, -1.25])

    single_point_reaction(theory=theory, reaction=reaction)

    assert reaction.properties["E_var"] == [-1.0, -1.25]
    assert reaction.properties["E_PT2_rest"] == [-0.01, -0.0125]
    assert reaction.properties["num_genCFGs"] == [10, 10]
    assert reaction.properties["num_selected_CFGs"] == [34, 34]
    assert reaction.properties["num_after_SD_CFGs"] == [90, 90]


@pytest.mark.usefixtures("fake_orca_dir")
def test_single_point_reaction_tolerates_an_orca_theory_without_ice_ci_properties():
    reaction = Reaction(fragments=labelled_fragments(2), stoichiometry=[-1, 1])
    theory = _IceCIORCATheory([-1.0, -1.25], with_properties=False)

    single_point_reaction(theory=theory, reaction=reaction)

    assert all(values == [] for values in reaction.properties.values())
    assert reaction.reaction_energy == pytest.approx(-0.25 * HARTREE_TO_EV)


def test_reaction_energy_requires_a_stoichiometry():
    with pytest.raises(InputError, match="stoichiometry list is required"):
        reaction_energy(list_of_energies=[-1.0, -1.25])


def test_reaction_energy_requires_energies_or_fragments():
    with pytest.raises(InputError, match="Provide either list_of_energies or list_of_fragments"):
        reaction_energy(stoichiometry=[-1, 1])


def test_reaction_energy_applies_a_correction_and_reports_the_error_against_a_reference(caplog):
    with caplog.at_level(logging.INFO, logger="openmmqmmm.singlepoint"):
        energy, error = reaction_energy(
            list_of_energies=[-1.0, -1.25],
            stoichiometry=[-1, 1],
            unit="kcal/mol",
            correction=0.1,
            reference=-90.0,
            label="corrected",
        )

    expected = -0.15 * HARTREE_TO_KCAL_PER_MOL
    assert energy == pytest.approx(expected)
    assert error == pytest.approx(expected + 90.0)
    assert f"Reaction_energy(corrected):  {energy} kcal/mol (Error: {error})" in caplog.text
    assert "User-correction was added" in caplog.text
    logged_correction = float(re.search(r"correction_in_unit in (\S+) kcal/mol", caplog.text).group(1))
    assert logged_correction == pytest.approx(0.1 * HARTREE_TO_KCAL_PER_MOL)

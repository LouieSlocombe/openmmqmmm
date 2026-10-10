"""Parsers and writers of openmmqmmm.orca with no fixture in test_orca_parsers.py."""

import logging
import re
from pathlib import Path

import numpy as np
import pytest

from openmmqmmm import Fragment, orca
from openmmqmmm.exceptions import ExternalProgramError

FIXTURES = Path(__file__).parent / "orca_outputs"
HF_ARGS = ("hf", ["H", "F"], [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], "! HF def2-SVP", "%scf maxiter 50 end", 0, 1)


def _section(text, start, stop):
    return text.split(start, 1)[1].split(stop, 1)[0]


def _raw_polarizability(text):
    block = _section(text, "The raw cartesian tensor (atomic units):\n", "diagonalized tensor:")
    return np.array([[float(value) for value in line.split()] for line in block.strip().splitlines()])


def _diagonalized_polarizability(text):
    line = text.split("diagonalized tensor: \n", 1)[1].splitlines()[0]
    return [float(value) for value in line.split()]


ORBITAL_ROW = re.compile(r"^\s+\d+\s+(\d\.\d+)\s+-?\d+\.\d+\s+-?\d+\.\d+\s*$", re.MULTILINE)


def _orbital_occupations(section):
    return [float(occ) for occ in ORBITAL_ROW.findall(section)]


def test_polarizability_tensor_matches_the_raw_tensor_in_the_output():
    text = (FIXTURES / "h2o_polar.out").read_text()

    tensor, diagonal = orca._grab_polarizability_tensor(FIXTURES / "h2o_polar.out")

    assert tensor == pytest.approx(_raw_polarizability(text))
    assert diagonal == pytest.approx(_diagonalized_polarizability(text))
    assert len(diagonal) == 3


def test_polarizability_diagonal_is_the_spectrum_of_the_raw_tensor():
    tensor, diagonal = orca._grab_polarizability_tensor(FIXTURES / "h2o_polar.out")

    assert tensor == pytest.approx(tensor.T, abs=1e-9)
    assert sorted(diagonal) == pytest.approx(np.linalg.eigvalsh(tensor), abs=1e-6)


def test_polarizability_tensor_is_empty_without_an_elprop_section():
    tensor, diagonal = orca._grab_polarizability_tensor(FIXTURES / "h2o_engrad.out")

    assert not tensor.any()
    assert diagonal == []


def test_fod_occupations_match_the_orbital_energy_table():
    text = (FIXTURES / "h2o_fod.out").read_text()
    expected = _orbital_occupations(_section(text, "ORBITAL ENERGIES", "MULLIKEN POPULATION ANALYSIS"))

    occupations = orca._grab_scf_fod_occupations(FIXTURES / "h2o_fod.out")

    assert occupations == pytest.approx(expected)
    assert len(occupations) == 18
    assert 0.0 < occupations[4] < 2.0, "smearing must leave the HOMO fractionally occupied"
    assert sum(occupations) == pytest.approx(10.0, abs=1e-3), "ten electrons in water"


def test_fod_occupations_of_an_open_shell_output_are_the_spin_up_block():
    text = (FIXTURES / "h2o_cation_charges.out").read_text()
    expected = _orbital_occupations(_section(text, "SPIN UP ORBITALS", "SPIN DOWN ORBITALS"))

    occupations = orca._grab_scf_fod_occupations(FIXTURES / "h2o_cation_charges.out")

    assert occupations == pytest.approx(expected)
    assert sum(occupations) == pytest.approx(5.0), "five alpha electrons in the water cation"


def test_fod_occupations_of_a_truncated_output_are_what_was_printed(tmp_path):
    outfile = tmp_path / "truncated.out"
    outfile.write_text(
        "ORBITAL ENERGIES\n----------------\n\n  NO   OCC          E(Eh)            E(eV) \n"
        "   0   2.0000     -18.879917      -513.7487 \n   1   1.9996      -0.261243        -7.1088 \n"
    )

    assert orca._grab_scf_fod_occupations(outfile) == [2.0, 1.9996]


def test_ice_configuration_counts_match_the_output():
    text = (FIXTURES / "h2o_ice.out").read_text()
    generators = int(re.findall(r"# of generator configurations\s+\.\.\.\s+(\d+)", text)[-1])
    after_sd = int(re.findall(r"# of configurations after S\+D\s+\.\.\.\s+(\d+)", text)[-1])
    selected = int(re.findall(r"# of configurations after Selection\s+\.\.\.\s+(\d+)", text)[-1])

    assert orca._grab_ice_wf_cfg_ci_size(FIXTURES / "h2o_ice.out") == (generators, selected, after_sd)
    assert generators < selected < after_sd


def test_ice_configuration_counts_are_zero_for_a_non_ice_output():
    assert orca._grab_ice_wf_cfg_ci_size(FIXTURES / "h2o_engrad.out") == (0, 0, 0)


def _broken_symmetry_output(tmp_path, numatoms):
    outfile = tmp_path / "bs.out"
    outfile.write_text(
        f"Number of atoms                             ...      {numatoms}\n"
        "WARNING: Broken symmetry calculations are performed\n"
    )
    return str(outfile)


def test_broken_symmetry_trim_keeps_the_last_table(tmp_path, caplog):
    outfile = _broken_symmetry_output(tmp_path, 2)
    with caplog.at_level(logging.INFO, logger="openmmqmmm.orca"):
        trimmed = orca._trim_to_broken_symmetry_solution([1.0, 2.0, 3.0, 4.0], outfile, "charges")

    assert trimmed == [3.0, 4.0]
    assert "Only taking BS-state charges" in caplog.text


def test_broken_symmetry_trim_leaves_a_single_table_alone(tmp_path):
    outfile = _broken_symmetry_output(tmp_path, 2)

    assert orca._trim_to_broken_symmetry_solution([1.0, 2.0], outfile, "charges") == [1.0, 2.0]


def test_trim_is_a_no_op_without_the_broken_symmetry_warning():
    values = [1.0, 2.0, 3.0, 4.0]

    assert orca._trim_to_broken_symmetry_solution(values, FIXTURES / "h2o_engrad.out", "charges") == values


@pytest.mark.parametrize("model", ["NPA", "nbo"])
def test_npa_charges_warn_about_the_nbo_executable(caplog, model):
    with caplog.at_level(logging.WARNING, logger="openmmqmmm.orca"):
        charges = orca.grab_orca_atom_charges(model, FIXTURES / "h2o_engrad.out")

    assert charges == []
    assert "NBOEXE" in caplog.text


def test_opt_finished_reports_the_cycle_count_of_a_real_optimization(caplog):
    text = (FIXTURES / "h2o_extopt.out").read_text()
    cycles = re.search(r"\(AFTER\s+(\d+) CYCLES\)", text).group(1)

    with caplog.at_level(logging.INFO, logger="openmmqmmm.orca"):
        converged = orca._check_orca_opt_finished(FIXTURES / "h2o_extopt.out")

    assert converged is True
    assert f"converged in {cycles} cycles" in caplog.text


def test_opt_finished_is_false_for_a_single_point_output():
    assert orca._check_orca_opt_finished(FIXTURES / "h2o_engrad.out") is False


OPT_OUTPUT = (
    "SCF CONVERGED AFTER   7 CYCLES\n"
    "THE OPTIMIZATION HAS CONVERGED\n"
    "***               (AFTER    3 CYCLES)               ***\n"
    "FINAL SINGLE POINT ENERGY      -1.250000000000\n"
    "TOTAL RUN TIME: 0 days 0 hours 0 minutes 0 seconds 10 msec\n"
)


@pytest.mark.parametrize(
    ("output", "message"),
    [
        ("SCF CONVERGED AFTER   7 CYCLES\n", "did not finish"),
        ("FINAL SINGLE POINT ENERGY -1.0\nTOTAL RUN TIME: 0\n", "failed to converge"),
        ("THE OPTIMIZATION HAS CONVERGED\nTOTAL RUN TIME: 0\n", "no final energy"),
    ],
)
def test_read_optimized_geometry_rejects_incomplete_outputs(output, message):
    Path("job.out").write_text(output)
    fragment = Fragment(elems=["H", "H"], coords=[[0, 0, 0], [0, 0, 0.74]], charge=0, mult=1)

    with pytest.raises(ExternalProgramError, match=message):
        orca._read_optimized_geometry(fragment, "job")


def test_read_optimized_geometry_updates_coordinates_and_energy():
    Path("job.out").write_text(OPT_OUTPUT)
    Path("job.xyz").write_text("2\nCoordinates from ORCA-job job E -1.25\nH 0.0 0.0 0.0\nH 0.0 0.0 0.8\n")
    fragment = Fragment(elems=["H", "H"], coords=[[0, 0, 0], [0, 0, 0.74]], charge=0, mult=1)

    energy = orca._read_optimized_geometry(fragment, "job")

    assert energy == -1.25
    assert fragment.energy == -1.25
    assert np.allclose(fragment.coords, [[0, 0, 0], [0, 0, 0.8]])


def test_final_energy_rejects_an_unconverged_wavefunction(tmp_path):
    outfile = tmp_path / "unconverged.out"
    outfile.write_text("FINAL SINGLE POINT ENERGY   -1.0  (Wavefunction not fully converged!)\n")

    with pytest.raises(ExternalProgramError, match="not fully converged"):
        orca.grab_orca_final_energy(outfile)


@pytest.mark.parametrize(
    ("line", "seconds"),
    [
        ("Sum of individual times         ....       0.527 sec", 0.527),
        ("SCF iterations                  ....       0.123 sec ( 95.0%)", 0.123),
        ("Total time                      ....       (1.5 sec)", 1.5),
        ("Sum of individual times         ....       n/a sec", None),
        ("sec is the first word here", None),
        ("no timing on this line", None),
    ],
)
def test_seconds_from_timing_line(line, seconds):
    assert orca._seconds_from_timing_line(line) == seconds


def test_print_gradient_in_orca_format_writes_the_engrad_layout():
    orca.print_gradient_in_orca_format(-1.5, [[0.1, 0.2, 0.3], [-0.1, -0.2, -0.3]], "job")

    assert Path("job_EXT.engrad").read_text() == (
        "#\n# Number of atoms\n#\n         2\n"
        "#\n# The current total energy in E\n#\n     -1.5\n"
        "#\n# The current gradient in Eh/Bohr\n#\n"
        "0.1\n0.2\n0.3\n-0.1\n-0.2\n-0.3\n"
        "#\n# The atomic numbers and current coordinates in Bohr\n#\n"
    )


def test_print_gradient_in_orca_format_honours_the_extra_basename():
    orca.print_gradient_in_orca_format(0.0, np.zeros((1, 3)), "job", extrabasename="")

    assert Path("job.engrad").read_text().splitlines()[11:14] == ["0.0", "0.0", "0.0"]


def test_create_orca_pcfile_writes_charge_then_coordinates():
    orca._create_orca_pcfile("field", [[0.0, 1.2, 2.0], [0.0, 1.85, 2.75]], [0.417, -0.417])

    assert Path("field.pc").read_text() == "2\n0.417 0.0 1.2 2.0\n-0.417 0.0 1.85 2.75\n"


def test_create_orca_input_writes_every_optional_block_in_order():
    orca._create_orca_input(
        *HF_ARGS,
        pcfile="hf.pc",
        grad=True,
        hessian=True,
        extraline="! TightSCF",
        moreadfile="start.gbw",
        propertyblock="%elprop Dipole true end\n",
        delta_scf_block="! DELTASCF\n%scf\n PMOM True\nend",
    )

    assert Path("hf.inp").read_text() == (
        "! HF def2-SVP\n! TightSCF\n! Engrad\n! Freq\n"
        '\n! MOREAD\n%moinp "start.gbw"\n%pointcharges "hf.pc"\n%scf maxiter 50 end\n'
        "\n! DELTASCF\n%scf\n PMOM True\nend\n"
        "*xyz 0 1\nH 0.0 0.0 0.0 \nF 0.0 0.0 1.0 \n*\n%elprop Dipole true end\n"
    )


@pytest.mark.parametrize(
    ("atomstoflip", "flipline"), [([1], "Flipspin 1"), ([0, 1], "Flipspin 0,1"), (1, "Flipspin 1")]
)
def test_create_orca_input_broken_symmetry_starts_from_the_high_spin_state(atomstoflip, flipline):
    orca._create_orca_input(*HF_ARGS, hs_mult=3, atomstoflip=atomstoflip)

    assert Path("hf.inp").read_text() == (
        f"! HF def2-SVP\n%scf maxiter 50 end\n%scf\n{flipline}\nFinalMs 0.0\nend  \n\n\n"
        "*xyz 0 3\nH 0.0 0.0 0.0 \nF 0.0 0.0 1.0 \n*\n"
    )


def test_create_orca_input_rohf_uhf_swap_appends_a_noiter_uhf_job():
    orca._create_orca_input("hf", *HF_ARGS[1:3], "! ROHF def2-SVP", "%scf maxiter 50 end", 1, 2, rohf_uhf_swap=True)

    assert Path("hf.inp").read_text() == (
        "! ROHF def2-SVP\n%scf maxiter 50 end\n\n\n*xyz 1 2\nH 0.0 0.0 0.0 \nF 0.0 0.0 1.0 \n*\n"
        "\n$new_job\n! UHF noiter  def2-SVP\n%scf maxiter 50 end\n* xyz 1 2\nH 0.0 0.0 0.0 \nF 0.0 0.0 1.0 \n*\n"
    )


def _coordinate_lines():
    return Path("hf.inp").read_text().split("*xyz 0 1\n", 1)[1].split("*\n")[0].splitlines()


def test_create_orca_input_extra_basis_atoms():
    orca._create_orca_input(*HF_ARGS, extrabasis="def2-TZVP", extrabasisatoms=[1])

    assert _coordinate_lines() == ["H 0.0 0.0 0.0 ", 'F 0.0 0.0 1.0 newgto "def2-TZVP" end']


def test_create_orca_input_atom_specific_basis_lines_follow_each_atom():
    basis = {("H", 0): ['newgto "cc-pVDZ" end\n'], ("F", 1): ['newgto "cc-pVTZ" end\n', 'newECP "none" end\n']}
    orca._create_orca_input(*HF_ARGS, atom_specific_basis_dict=basis)

    assert _coordinate_lines() == [
        "H 0.0 0.0 0.0 ",
        'newgto "cc-pVDZ" end',
        "F 0.0 0.0 1.0 ",
        'newgto "cc-pVTZ" end',
        'newECP "none" end',
    ]


def test_create_orca_input_ghost_and_dummy_atoms():
    orca._create_orca_input(*HF_ARGS, ghostatoms=[0], dummyatoms=[1])

    assert _coordinate_lines() == ["H: 0.0 0.0 0.0 ", "DA 0.0 0.0 1.0 "]


def test_create_orca_input_fragment_tags():
    orca._create_orca_input(*HF_ARGS, fragment_indices=[[1], [0]])

    assert _coordinate_lines() == ["H(2) 0.0 0.0 0.0 ", "F(1) 0.0 0.0 1.0 "]

"""ORCATheory construction, one-shot option handling and file bookkeeping, without launching ORCA."""

import logging
import os
import re
import shutil
from pathlib import Path

import numpy as np
import pytest

from openmmqmmm import Fragment, ORCATheory, orca
from openmmqmmm.exceptions import ExternalProgramError, InputError

FIXTURES = Path(__file__).parent / "orca_outputs"
# CODATA 2018: Boltzmann constant in Hartree per kelvin.
BOLTZMANN_HARTREE_PER_K = 3.166811563e-6

pytestmark = pytest.mark.usefixtures("fake_orca_dir")


def _theory(**options):
    options.setdefault("orcasimpleinput", "! HF def2-SVP")
    return ORCATheory(**options)


def _mulliken_charges(text):
    table = text.split("MULLIKEN ATOMIC CHARGES\n", 1)[1].split("Sum of atomic charges", 1)[0]
    return [float(value) for value in re.findall(r"^\s*\d+\s+[A-Za-z]+\s*:\s+(-?\d+\.\d+)\s*$", table, re.MULTILINE)]


@pytest.mark.parametrize("simpleinput", ["! HF def2-SVP Opt", "! HF def2-SVP freq"])
def test_init_rejects_job_directives_in_the_simple_input(simpleinput):
    with pytest.raises(InputError, match="job-directives"):
        ORCATheory(orcasimpleinput=simpleinput)


def test_init_requires_a_bang_line():
    with pytest.raises(InputError, match="'!'"):
        ORCATheory(orcasimpleinput="HF def2-SVP")


def test_save_output_with_label_requires_a_label():
    with pytest.raises(InputError, match="label"):
        _theory(save_output_with_label=True, label=None)


def test_autostart_false_adds_the_noautostart_keyword():
    assert _theory(autostart=False, extraline="! TightSCF").extraline == "! TightSCF\n! Noautostart\n"
    assert _theory(autostart=True, extraline="! TightSCF").extraline == "! TightSCF"


def test_first_iteration_input_defaults_to_empty():
    assert _theory().first_iteration_input == ""
    assert _theory(first_iteration_input="! SlowConv").first_iteration_input == "! SlowConv"


def test_atoms_to_flip_must_be_a_sequence():
    with pytest.raises(InputError, match="list of integers"):
        _theory(atomstoflip=2)


def test_broken_symmetry_adds_uks_when_missing(caplog):
    with caplog.at_level(logging.WARNING, logger="openmmqmmm.orca"):
        theory = _theory(orcasimpleinput="! BP86 def2-SVP", brokensym=True, hs_mult=3, atomstoflip=[0])

    assert theory.orcasimpleinput == "! BP86 def2-SVP UKS"
    assert theory.atomstoflip == [0]
    assert "UKS/UHF keyword not present" in caplog.text


def test_broken_symmetry_keeps_an_existing_uhf_keyword():
    assert _theory(orcasimpleinput="! UHF def2-SVP", brokensym=True).orcasimpleinput == "! UHF def2-SVP"


def test_delta_scf_requires_a_configuration_line():
    with pytest.raises(InputError, match="deltaSCF_confline"):
        _theory(delta_scf=True)


def test_delta_scf_turns_on_population_printing():
    theory = _theory(delta_scf=True, delta_scf_confline="SetAlpha 4 5 end", print_population_analysis=False)

    assert theory.print_population_analysis is True


def test_parallel_theory_checks_openmpi_once(monkeypatch):
    calls = []
    monkeypatch.setattr("openmmqmmm.parallel.check_openmpi", lambda: calls.append(True))

    assert _theory(numcores=1).numcores == 1
    assert calls == []
    assert _theory(numcores=4).numcores == 4
    assert calls == [True]


def test_set_numcores():
    theory = _theory()
    theory.set_numcores(3)

    assert theory.numcores == 3


def test_cleanup_removes_only_the_scratch_files_of_this_filename():
    theory = _theory(filename="job")
    scratch = ["job.gbw", "job.densities", "job.ges", "job.prop", "job.uco", "job.property.txt", "job.inp"]
    scratch += ["job.engrad", "job.cis", "job_last.out", "job.xyz", "job.scfp.tmp", "job_bla.tmp"]
    kept = ["job.out", "job_run1.out", "other.gbw", "job.hess", "fragment.frag"]
    for name in scratch + kept:
        Path(name).write_text("x")

    theory.cleanup()

    assert all(not Path(name).exists() for name in scratch)
    assert all(Path(name).exists() for name in kept)
    theory.cleanup()


def test_nmf_requires_sigma():
    with pytest.raises(InputError, match="NMF_sigma"):
        _theory(nmf=True)


def test_nmf_adds_fermi_smearing_block():
    theory = _theory(nmf=True, nmf_sigma=0.02, orcablocks="%scf maxiter 50 end")

    smeartemp = float(re.search(r"smeartemp (\S+)", theory.orcablocks).group(1))
    assert theory.orcablocks.startswith("%scf maxiter 50 end\n%scf\nfracocc true\nsmeartemp ")
    assert theory.orcablocks.endswith("\nend\n")
    assert smeartemp == pytest.approx(0.02 / BOLTZMANN_HARTREE_PER_K, rel=1e-6)


def test_tddft_block_is_added_once():
    theory = _theory(tddft=True, tddft_roots=3, follow_root=2)
    assert theory.orcablocks == "\n%tddft\nnroots 3\nIRoot 2\nend\n"

    user_block = "%tddft nroots 8 end"
    assert _theory(tddft=True, orcablocks=user_block).orcablocks == user_block


def test_cpcm_radii_block():
    theory = _theory(cpcm_radii=[1.5, 1.2])

    assert theory.orcablocks == "%cpcm\nAtomRadii(0,  1.5)\nAtomRadii(1,  1.2)\nend\n"


def test_xdm_requests_the_wfx_file():
    assert _theory(xdm=True).orcasimpleinput == "! HF def2-SVP AIM"
    assert _theory(xdm=False).xdm is False


def test_basis_per_element_block_is_appended_once():
    theory = _theory(basis_per_element={"O": "def2-TZVP", "H": "def2-SVP"}, orcablocks="%scf maxiter 50 end")

    theory._append_basis_blocks()
    theory._append_basis_blocks()

    assert theory.orcablocks == (
        '%scf maxiter 50 end\n%basis\nnewgto O "def2-TZVP" end\nnewgto H "def2-SVP" end\n\nend'
    )


def test_ecp_block_is_appended_once():
    theory = _theory(ecp_dict={"I": ['newECP I "def2-ECP" end\n', 'newgto I "def2-SVP" end\n']})

    theory._append_basis_blocks()
    theory._append_basis_blocks()

    assert theory.orcablocks == '\n%basis\nnewECP I "def2-ECP" end\nnewgto I "def2-SVP" end\n\nend'


def test_reset_names_the_gbw_file_of_the_rohf_uhf_swap_job():
    theory = _theory(rohf_uhf_swap=True, filename="swap")
    theory._reset_one_shot_options()
    assert theory.gbwfile == "swap_job2.gbw"

    theory = _theory(filename="plain")
    theory._reset_one_shot_options()
    assert theory.gbwfile == "plain.gbw"


def test_reset_turns_off_broken_symmetry_after_one_run():
    theory = _theory(orcasimpleinput="! UKS BP86 def2-SVP", brokensym=True, hs_mult=3, atomstoflip=[0])

    theory._reset_one_shot_options()

    assert theory.brokensym is False


def test_reset_turns_off_delta_scf_and_pins_the_scf_solver():
    theory = _theory(delta_scf=True, delta_scf_confline="SetAlpha 4 5 end")

    theory._reset_one_shot_options()

    assert theory.delta_scf is False
    assert theory.orcasimpleinput == "! HF def2-SVP nososcf nodamp nolshift"
    theory._reset_one_shot_options()
    assert theory.orcasimpleinput == "! HF def2-SVP nososcf nodamp nolshift"


def test_reset_keeps_delta_scf_when_asked():
    theory = _theory(delta_scf=True, delta_scf_confline="SetAlpha 4 5 end", delta_scf_turn_off_automatically=False)

    theory._reset_one_shot_options()

    assert theory.delta_scf is True
    assert theory.orcasimpleinput == "! HF def2-SVP"


@pytest.mark.parametrize(("always", "expected"), [(False, None), (True, "start.gbw")])
def test_reset_drops_moreadfile_unless_always(always, expected):
    theory = _theory(moreadfile="start.gbw", moreadfile_always=always)

    theory._reset_one_shot_options()

    assert theory.moreadfile == expected


def test_save_output_copies_keeps_labelled_numbered_and_last_outputs():
    theory = _theory(filename="job", label="water", save_output_with_label=True, keep_each_run_output=True)
    theory.runcalls = 2
    Path("job.out").write_text("output text\n")

    theory._save_output_copies(0, 1)

    for copy in ("job_water_0_1.out", "job_run2.out", "job_last.out"):
        assert Path(copy).read_text() == "output text\n"
    assert theory.path_to_last_gbwfile_used == f"{os.getcwd()}/job.gbw"


def test_save_output_copies_only_the_last_output_by_default():
    theory = _theory(filename="job")
    Path("job.out").write_text("output text\n")

    theory._save_output_copies(0, 1)

    assert sorted(os.listdir(".")) == ["fake_orca", "job.out", "job_last.out"] or sorted(
        name for name in os.listdir(".") if name.startswith("job")
    ) == ["job.out", "job_last.out"]


def test_get_atomic_charges_reads_the_mulliken_table():
    shutil.copy(FIXTURES / "h2o_engrad.out", "orca.out")
    theory = _theory()

    charges = theory.get_atomic_charges()

    assert charges == pytest.approx(_mulliken_charges((FIXTURES / "h2o_engrad.out").read_text()))
    assert theory.properties["Mulliken_charges"] == pytest.approx(charges)


def test_get_atomic_charges_without_output_or_table():
    theory = _theory()
    with pytest.raises(InputError, match="No readable ORCA output"):
        theory.get_atomic_charges()

    Path("orca.out").write_text("no population analysis here\n")
    with pytest.raises(InputError, match="finite Mulliken"):
        theory.get_atomic_charges()


def test_population_log_closed_shell_lists_charges(caplog):
    shutil.copy(FIXTURES / "h2o_engrad.out", "orca.out")
    expected = _mulliken_charges((FIXTURES / "h2o_engrad.out").read_text())
    theory = _theory()

    with caplog.at_level(logging.INFO, logger="openmmqmmm.orca"):
        theory._log_population_analysis(["O", "H", "H"])

    assert "Spinpop" not in caplog.text
    assert f"0  O :    {expected[0]:.4f}" in caplog.text
    assert theory.properties["Mulliken_spinpops"] == []
    assert theory.properties["Mulliken_charges"] == pytest.approx(expected)


def test_population_log_open_shell_lists_spin_populations(caplog):
    shutil.copy(FIXTURES / "h2o_cation_charges.out", "orca.out")
    theory = _theory()

    with caplog.at_level(logging.INFO, logger="openmmqmmm.orca"):
        theory._log_population_analysis(["O", "H", "H"])

    assert "Charge    Spinpop" in caplog.text
    assert sum(theory.properties["Mulliken_spinpops"]) == pytest.approx(1.0, abs=1e-5)
    assert len(theory.properties["Mulliken_charges"]) == 3


def test_population_log_warns_when_no_table_exists(caplog):
    Path("orca.out").write_text("nothing\n")
    theory = _theory()

    with caplog.at_level(logging.WARNING, logger="openmmqmmm.orca"):
        theory._log_population_analysis(["O", "H", "H"])

    assert "No charges or spinpops were found" in caplog.text


def test_optional_properties_collect_ice_ci_sizes():
    shutil.copy(FIXTURES / "h2o_ice.out", "orca.out")
    text = (FIXTURES / "h2o_ice.out").read_text()
    theory = _theory()
    theory.energy = -75.0

    theory._collect_optional_properties("orca.out")

    assert theory.properties["E_var"] == -75.0
    assert theory.properties["E_PT2_rest"] == float(re.search(r"'rest' energy.*\.\.\.\s+(\S+)", text).group(1))
    assert theory.properties["num_genCFGs"] == int(re.findall(r"generator configurations\s+\.\.\.\s+(\d+)", text)[-1])
    assert theory.properties["num_after_SD_CFGs"] == int(re.findall(r"after S\+D\s+\.\.\.\s+(\d+)", text)[-1])
    assert theory.properties["num_selected_CFGs"] == int(re.findall(r"after Selection\s+\.\.\.\s+(\d+)", text)[-1])


def test_optional_properties_skip_ice_ci_for_a_plain_scf_output():
    shutil.copy(FIXTURES / "h2o_engrad.out", "orca.out")
    theory = _theory()
    theory.energy = -75.0

    theory._collect_optional_properties("orca.out")

    assert theory.properties == {}


def _fermi_entropy_energy(occupations, sigma):
    """Ec = 2 sigma sum[f ln f + (1-f) ln(1-f)] over fractionally occupied spatial orbitals."""
    f = np.asarray(occupations) / 2.0
    f = f[(f > 0) & (f < 1)]
    return 2.0 * sigma * np.sum(f * np.log(f) + (1 - f) * np.log(1 - f))


def test_optional_properties_add_the_nmf_entropy_correction():
    shutil.copy(FIXTURES / "h2o_fod.out", "orca.out")
    table = (FIXTURES / "h2o_fod.out").read_text().split("ORBITAL ENERGIES", 1)[1].split("MULLIKEN", 1)[0]
    occupations = [float(v) for v in re.findall(r"^\s+\d+\s+(\d\.\d+)\s+-?\d+\.\d+\s+-?\d+\.\d+", table, re.MULTILINE)]
    theory = _theory(nmf=True, nmf_sigma=0.05)
    theory.energy = -76.0

    theory._collect_optional_properties("orca.out")

    expected_ec = _fermi_entropy_energy(occupations, 0.05)
    assert expected_ec < 0.0
    assert theory.properties["NMF_occupations"] == pytest.approx(occupations)
    assert theory.properties["E_NMF"] == -76.0
    assert theory.properties["NMF_Ec"] == pytest.approx(expected_ec)
    assert theory.energy == pytest.approx(-76.0 + expected_ec)


def test_optional_properties_collect_tddft_spectrum():
    shutil.copy(FIXTURES / "h2o_tddft.out", "orca.out")
    text = (FIXTURES / "h2o_tddft.out").read_text()
    energies = [float(v) for v in re.findall(r"^STATE\s+\d+:\s+E=\s+\S+ au\s+(\S+) eV", text, re.MULTILINE)]
    spectrum = text.split("ELECTRIC DIPOLE MOMENTS", 1)[1].split("VELOCITY DIPOLE MOMENTS", 1)[0]
    intensities = [float(v) for v in re.findall(r"->\s+\S+\s+\S+\s+\S+\s+\S+\s+(\S+)", spectrum)]
    theory = _theory(tddft=True)

    theory._collect_optional_properties("orca.out")

    assert len(energies) == 3
    assert theory.properties["TDDFT_transition_energies"] == pytest.approx(energies)
    assert theory.properties["TDDFT_transition_intensities"] == pytest.approx(intensities)


def test_dipole_moment_from_last_output():
    shutil.copy(FIXTURES / "h2o_engrad.out", "orca.out")
    match = re.search(r"Total Dipole Moment\s+:\s+(\S+)\s+(\S+)\s+(\S+)", (FIXTURES / "h2o_engrad.out").read_text())

    assert _theory().get_dipole_moment() == pytest.approx([float(v) for v in match.groups()])


def test_polarizability_tensor_from_last_output():
    shutil.copy(FIXTURES / "h2o_polar.out", "orca.out")
    block = (FIXTURES / "h2o_polar.out").read_text().split("raw cartesian tensor (atomic units):\n", 1)[1]
    expected = [[float(v) for v in line.split()] for line in block.splitlines()[:3]]

    assert _theory().get_polarizability_tensor() == pytest.approx(np.array(expected))


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    return work


def test_orbital_guess_keeps_a_moreadfile_present_in_cwd(workdir, caplog):
    (workdir / "start.gbw").write_text("orbitals")
    theory = _theory(moreadfile="start.gbw")

    with caplog.at_level(logging.INFO, logger="openmmqmmm.orca"):
        theory._prepare_orbital_guess()

    assert "File exists in current directory" in caplog.text
    assert (workdir / "start.gbw").read_text() == "orbitals"


def test_orbital_guess_rejects_a_missing_absolute_moreadfile(workdir):
    theory = _theory(moreadfile=str(workdir / "missing.gbw"))

    with pytest.raises(InputError, match="absolute path that does not exist"):
        theory._prepare_orbital_guess()


def test_orbital_guess_copies_a_relative_moreadfile_from_the_parent(workdir):
    (workdir.parent / "start.gbw").write_text("parent orbitals")
    theory = _theory(moreadfile="start.gbw")

    theory._prepare_orbital_guess()

    assert (workdir / "start.gbw").read_text() == "parent orbitals"


def test_orbital_guess_tolerates_a_relative_moreadfile_found_nowhere(workdir, caplog):
    theory = _theory(moreadfile="start.gbw")

    with caplog.at_level(logging.INFO, logger="openmmqmmm.orca"):
        theory._prepare_orbital_guess()

    assert "File does not exist in current directory" in caplog.text
    assert not (workdir / "start.gbw").exists()


@pytest.mark.parametrize(("autostart", "message"), [(True, "ORCA will read GBW-file"), (False, "ORCA will ignore")])
def test_orbital_guess_reports_a_gbw_file_already_in_cwd(workdir, caplog, autostart, message):
    (workdir / "orca.gbw").write_text("orbitals")
    theory = _theory(autostart=autostart)

    with caplog.at_level(logging.INFO, logger="openmmqmmm.orca"):
        theory._prepare_orbital_guess()

    assert "orca.gbw is present" in caplog.text
    assert message in caplog.text


def test_orbital_guess_copies_the_parent_gbw_file(workdir):
    (workdir.parent / "orca.gbw").write_text("parent orbitals")

    _theory()._prepare_orbital_guess()

    assert (workdir / "orca.gbw").read_text() == "parent orbitals"


def test_orbital_guess_lets_orca_guess_when_nothing_is_found(workdir, caplog):
    with caplog.at_level(logging.INFO, logger="openmmqmmm.orca"):
        _theory()._prepare_orbital_guess()

    assert "ORCA will guess new orbitals" in caplog.text
    assert not (workdir / "orca.gbw").exists()


def test_orbital_guess_copies_the_last_gbw_file_used(workdir, tmp_path):
    previous = tmp_path / "previous" / "orca.gbw"
    previous.parent.mkdir()
    previous.write_text("previous orbitals")
    theory = _theory()
    theory.path_to_last_gbwfile_used = str(previous)

    theory._prepare_orbital_guess()

    assert (workdir / "orca.gbw").read_text() == "previous orbitals"


def test_orbital_guess_survives_a_deleted_last_gbw_file(workdir, tmp_path, caplog):
    theory = _theory(autostart=False)
    theory.path_to_last_gbwfile_used = str(tmp_path / "gone.gbw")

    with caplog.at_level(logging.INFO, logger="openmmqmmm.orca"):
        theory._prepare_orbital_guess()

    assert "File was not found" in caplog.text
    assert "Autostart option is False" in caplog.text
    assert not (workdir / "orca.gbw").exists()


def test_opt_requires_a_fragment():
    with pytest.raises(InputError, match="No fragment"):
        _theory().opt()


def test_opt_requires_a_charge_and_multiplicity():
    fragment = Fragment(elems=["H", "H"], coords=[[0, 0, 0], [0, 0, 0.74]])

    with pytest.raises(InputError, match="charge/mult"):
        _theory().opt(fragment)


def test_opt_reads_the_optimized_geometry_back(monkeypatch, caplog):
    def fake_orca(orcadir, inpfile, **kwargs):
        Path("orca.out").write_text(
            "SCF CONVERGED AFTER   7 CYCLES\nTHE OPTIMIZATION HAS CONVERGED\n"
            "***               (AFTER    3 CYCLES)               ***\n"
            "FINAL SINGLE POINT ENERGY      -1.250000000000\n"
            "TOTAL RUN TIME: 0 days 0 hours 0 minutes 0 seconds 10 msec\n"
        )
        Path("orca.xyz").write_text("2\nCoordinates from ORCA-job orca E -1.25\nH 0.0 0.0 0.0\nH 0.0 0.0 0.8\n")

    monkeypatch.setattr("openmmqmmm.orca._run_orca_sp_parallel", fake_orca)
    fragment = Fragment(elems=["H", "H"], coords=[[0, 0, 0], [0, 0, 0.74]], charge=0, mult=1)
    theory = _theory()

    with caplog.at_level(logging.INFO, logger="openmmqmmm.orca"):
        energy = theory.opt(fragment)

    assert energy == theory.energy == fragment.energy == -1.25
    assert np.allclose(fragment.coords, [[0, 0, 0], [0, 0, 0.8]])
    assert "converged in 3 cycles" in caplog.text
    assert Path("fragment_optimized.frag").exists()
    assert Path("Fragment-optimized.xyz").read_text().splitlines()[3].split()[1:] == ["0.00000000"] * 2 + ["0.80000000"]


def test_find_orca_expands_the_home_directory(tmp_path, monkeypatch, make_fake_orca_install):
    make_fake_orca_install(tmp_path / "home_orca")
    monkeypatch.setenv("HOME", str(tmp_path))

    assert orca.find_orca("~/home_orca") == str(tmp_path / "home_orca")


def test_find_orca_rejects_a_directory_without_helper_binaries(tmp_path, make_fake_orca_install):
    bare = make_fake_orca_install(tmp_path / "bare", with_helpers=False)

    with pytest.raises(ExternalProgramError, match="orcadir argument points at"):
        orca.find_orca(bare)
    assert orca.find_orca(bare, required=False) is None


def test_find_orca_rejects_a_binary_that_is_not_orca(tmp_path, make_fake_orca_install, caplog):
    impostor = make_fake_orca_install(tmp_path / "impostor", output="Screen reader started")

    with caplog.at_level(logging.INFO, logger="openmmqmmm.orca"):
        assert orca.find_orca(impostor, required=False) is None

    assert "does not behave like the ORCA quantum chemistry program" in caplog.text


def test_orca_binary_runs_reports_an_unexecutable_binary(tmp_path, caplog):
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "orca").write_text("not executable")

    with caplog.at_level(logging.INFO, logger="openmmqmmm.orca"):
        assert orca._orca_binary_runs(broken) is False

    assert "could not be executed" in caplog.text


def test_find_orca_searches_path_when_no_directory_is_given(tmp_path, monkeypatch, make_fake_orca_install, caplog):
    monkeypatch.delenv("OPENMMQMMM_ORCADIR")
    installed = make_fake_orca_install(tmp_path / "in_path")
    monkeypatch.setenv("PATH", str(installed))

    with caplog.at_level(logging.INFO, logger="openmmqmmm.orca"):
        found = orca.find_orca()

    assert Path(found) == Path(os.path.realpath(installed))
    assert "found via PATH" in caplog.text


def test_find_orca_ignores_a_non_orca_program_in_path(tmp_path, monkeypatch, make_fake_orca_install, caplog):
    monkeypatch.delenv("OPENMMQMMM_ORCADIR")
    monkeypatch.setenv("PATH", str(make_fake_orca_install(tmp_path / "screen_reader", with_helpers=False)))

    with caplog.at_level(logging.INFO, logger="openmmqmmm.orca"):
        assert orca.find_orca(required=False) is None
    assert "ignoring" in caplog.text

    with pytest.raises(ExternalProgramError, match="Found no working ORCA installation"):
        orca.find_orca()

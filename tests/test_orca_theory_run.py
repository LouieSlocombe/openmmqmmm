"""ORCATheory.run option branches, driven by a fake orca binary that replays committed ORCA 6.1.1 output."""

import logging
import os
import re
import stat
from pathlib import Path

import numpy as np
import pytest

from openmmqmmm import Fragment, ORCATheory, ZeroTheory, orca, orca_external_optimizer
from openmmqmmm.exceptions import ExternalProgramError, InputError

FIXTURES = Path(__file__).parent / "orca_outputs"
PROBE_OUTPUT = "This program requires the name of a parameterfile"
WATER_ELEMS = ["O", "H", "H"]
WATER_COORDS = [[0.0, 0.0, 0.1173], [0.0, 0.7572, -0.4692], [0.0, -0.7572, -0.4692]]
WATER_BLOCK = "*xyz 0 1\nO 0.0 0.0 0.1173 \nH 0.0 0.7572 -0.4692 \nH 0.0 -0.7572 -0.4692 \n*\n"
# The point charges of h2o_pc.pc, so the replayed .pcgrad belongs to these MM atoms.
MM_COORDS = [[0.0, 1.2, 2.0], [0.0, 1.85, 2.75]]
MM_CHARGES = [0.417, -0.417]


def _install_fake_orca(directory, stem, exit_code, **sources):
    directory.mkdir(parents=True, exist_ok=True)
    lines = [
        "#!/bin/sh",
        f"if [ \"$#\" -eq 0 ]; then echo '{PROBE_OUTPUT}'; exit 2; fi",
        'base="${1%.inp}"',
        'printf "%s\\n" "$@" > "$base.args"',
    ]
    if stem is None:
        lines.append('echo "ORCA finished by error termination in SCF"')
    else:
        lines.append(f'cat "{FIXTURES / stem}.out"')
        for ext in ("engrad", "pcgrad", "hess"):
            source = sources.get(ext, FIXTURES / f"{stem}.{ext}")
            if source.exists():
                lines.append(f'cp "{source}" "$base.{ext}"')
    lines.append(f"exit {exit_code}")
    binary = directory / "orca"
    binary.write_text("\n".join(lines) + "\n")
    binary.chmod(binary.stat().st_mode | stat.S_IXUSR)
    (directory / "orca_scf").write_text("")
    return str(directory)


@pytest.fixture
def fake_orca(tmp_path):
    def make(stem="h2o_engrad", exit_code=0, **sources):
        return _install_fake_orca(tmp_path / f"fake_{stem}_{exit_code}", stem, exit_code, **sources)

    return make


def _theory(orcadir, **options):
    options.setdefault("orcasimpleinput", "! HF def2-SVP")
    return ORCATheory(orcadir=orcadir, **options)


def _run(theory, **options):
    options = {"current_coords": WATER_COORDS, "elems": WATER_ELEMS, "charge": 0, "mult": 1} | options
    return theory.run(**options)


def _final_energy(stem):
    return float(re.findall(r"FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+)", (FIXTURES / f"{stem}.out").read_text())[-1])


def _engrad_gradient(stem):
    lines = (FIXTURES / f"{stem}.engrad").read_text().splitlines()
    start = lines.index("# The current gradient in Eh/bohr") + 2
    return np.array([float(line) for line in lines[start : start + 9]]).reshape(3, 3)


def _pc_gradient(stem):
    rows = (FIXTURES / f"{stem}.pcgrad").read_text().splitlines()[1:]
    return np.array([[float(value) for value in row.split()] for row in rows])


def _hessian(stem):
    lines = (FIXTURES / f"{stem}.hess").read_text().splitlines()
    position = lines.index("$hessian") + 1
    dimension = int(lines[position])
    matrix = np.zeros((dimension, dimension))
    position += 1
    for block_start in range(0, dimension, 5):
        columns = [int(column) for column in lines[position].split()]
        assert columns[0] == block_start
        position += 1
        for _ in range(dimension):
            fields = lines[position].split()
            matrix[int(fields[0]), columns] = [float(value) for value in fields[1:]]
            position += 1
    return matrix


def _ir_intensities(stem):
    lines = (FIXTURES / f"{stem}.hess").read_text().splitlines()
    position = lines.index("$ir_spectrum") + 2
    return [float(line.split()[2]) for line in lines[position : position + 9]]


def _mulliken_charges(stem):
    table = (FIXTURES / f"{stem}.out").read_text().split("MULLIKEN ATOMIC CHARGES\n", 1)[1].split("Sum of", 1)[0]
    return [float(value) for value in re.findall(r":\s+(-?\d+\.\d+)", table)]


def _coordinate_lines(inpfile="orca.inp"):
    body = Path(inpfile).read_text().split("*xyz ", 1)[1].split("\n", 1)[1]
    return body.split("*\n", 1)[0].splitlines()


def test_run_replays_energy_and_gradient_and_writes_the_input(fake_orca):
    theory = _theory(fake_orca())

    energy, gradient = _run(theory, grad=True)

    assert energy == theory.energy == _final_energy("h2o_engrad")
    assert gradient == pytest.approx(_engrad_gradient("h2o_engrad"))
    assert Path("orca.inp").read_text() == "! HF def2-SVP\n\n! Engrad\n\n\n" + WATER_BLOCK
    assert Path("orca.args").read_text() == "orca.inp\n--bind-to none\n"
    assert Path("orca_last.out").read_text() == Path("orca.out").read_text()
    assert theory.path_to_last_gbwfile_used == f"{os.getcwd()}/orca.gbw"
    assert theory.runcalls == 1


def test_run_without_gradient_returns_only_the_energy(fake_orca):
    theory = _theory(fake_orca(), bind_to_core_option=False)

    energy = _run(theory)

    assert energy == _final_energy("h2o_engrad")
    assert "! Engrad" not in Path("orca.inp").read_text()
    assert not theory.grad.any()
    assert Path("orca.args").read_text() == "orca.inp\n"


def test_run_with_point_charges_returns_the_point_charge_gradient(fake_orca):
    # The point-charge fixture carries no .engrad; the gradient parser is exercised with the HF one.
    theory = _theory(fake_orca("h2o_pc", engrad=FIXTURES / "h2o_engrad.engrad"), orcasimpleinput="! BP86 def2-SVP")

    energy, gradient, pc_gradient = _run(theory, grad=True, pc=True, current_mm_coords=MM_COORDS, mm_charges=MM_CHARGES)

    assert energy == _final_energy("h2o_pc")
    assert gradient == pytest.approx(_engrad_gradient("h2o_engrad"))
    assert pc_gradient == pytest.approx(_pc_gradient("h2o_pc"))
    assert Path("orca.pc").read_text() == "2\n0.417 0.0 1.2 2.0\n-0.417 0.0 1.85 2.75\n"
    assert '%pointcharges "orca.pc"\n' in Path("orca.inp").read_text()


def test_run_with_hessian_reads_the_hess_file(fake_orca):
    theory = _theory(fake_orca("h2o_freq"))

    energy = _run(theory, hessian=True)

    assert energy == _final_energy("h2o_freq")
    assert "! Freq\n" in Path("orca.inp").read_text()
    assert theory.hessian == pytest.approx(_hessian("h2o_freq"))
    assert theory.ir_intensities == pytest.approx(_ir_intensities("h2o_freq"))


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"current_coords": None}, "No current_coords"),
        ({"charge": None}, "charge and mult"),
        ({"elems": None}, "No elems"),
    ],
)
def test_run_validates_its_arguments(fake_orca, overrides, message):
    with pytest.raises(InputError, match=message):
        _run(_theory(fake_orca()), **overrides)


def test_run_with_several_cores_inserts_a_pal_block(fake_orca, monkeypatch):
    monkeypatch.setattr("openmmqmmm.parallel.check_openmpi", lambda: None)
    theory = _theory(fake_orca(), numcores=2)

    _run(theory, grad=True)

    assert Path("orca.inp").read_text() == "! HF def2-SVP\n%pal \nnprocs 2\nend\n\n! Engrad\n\n\n" + WATER_BLOCK


def test_run_rejects_an_output_without_normal_termination(fake_orca):
    theory = _theory(fake_orca(None, exit_code=0))

    with pytest.raises(ExternalProgramError, match="did not terminate normally"):
        _run(theory)


def test_run_reports_a_failed_orca_process(fake_orca, caplog):
    theory = _theory(fake_orca(None, exit_code=1))

    with caplog.at_level(logging.ERROR, logger="openmmqmmm.orca"), pytest.raises(ExternalProgramError, match="failed"):
        _run(theory)

    assert "error termination in SCF" in caplog.text


def test_ignored_orca_error_without_energy_returns_zero(fake_orca, caplog):
    theory = _theory(fake_orca(None, exit_code=1), ignore_orca_error=True)

    with caplog.at_level(logging.INFO, logger="openmmqmmm.orca"):
        energy = _run(theory)

    assert energy == 0.0
    assert "ignored by user-input" in caplog.text


def test_ignored_orca_error_with_an_energy_returns_it(fake_orca):
    theory = _theory(fake_orca(exit_code=1), ignore_orca_error=True)

    assert _run(theory) == _final_energy("h2o_engrad")


def test_run_maps_qm_region_indices_for_extra_basis_and_fragments(fake_orca, caplog):
    theory = _theory(fake_orca(), extrabasisatoms=[5], extrabasis="def2-TZVP", fragment_indices=[[3, 5], [7]])
    theory.qmatoms = [3, 5, 7]

    with caplog.at_level(logging.INFO, logger="openmmqmmm.orca"):
        _run(theory)

    assert _coordinate_lines() == [
        "O(1) 0.0 0.0 0.1173 ",
        'H 0.0 0.7572 -0.4692 newgto "def2-TZVP" end',
        "H(2) 0.0 -0.7572 -0.4692 ",
    ]
    assert "Using extra basis (def2-TZVP) on QM-region indices : [1]" in caplog.text
    assert "List of fragment indices defined: [[0, 1], [2]]" in caplog.text


def test_run_rejects_flip_atoms_outside_the_qm_region(fake_orca):
    theory = _theory(fake_orca(), orcasimpleinput="! UKS BP86 def2-SVP", brokensym=True, hs_mult=3, atomstoflip=[9])
    theory.qmatoms = [3, 5, 7]

    with pytest.raises(InputError, match="not all in QM-region"):
        _run(theory)


def test_broken_symmetry_requires_the_high_spin_multiplicity(fake_orca):
    theory = _theory(fake_orca(), orcasimpleinput="! UKS BP86 def2-SVP", brokensym=True, atomstoflip=[0])

    with pytest.raises(InputError, match="HSmult"):
        _run(theory)


def test_broken_symmetry_requires_atoms_to_flip(fake_orca):
    theory = _theory(fake_orca(), orcasimpleinput="! UKS BP86 def2-SVP", brokensym=True, hs_mult=3)

    with pytest.raises(InputError, match="atomstoflip"):
        _run(theory)


def test_broken_symmetry_flips_spins_once(fake_orca, caplog):
    theory = _theory(fake_orca(), orcasimpleinput="! UKS BP86 def2-SVP", brokensym=True, hs_mult=3, atomstoflip=[0])

    with caplog.at_level(logging.INFO, logger="openmmqmmm.orca"):
        _run(theory)
    first_input = Path("orca.inp").read_text()
    _run(theory)
    second_input = Path("orca.inp").read_text()

    assert "%scf\nFlipspin 0\nFinalMs 0.0\nend  \n" in first_input
    assert "*xyz 0 3\n" in first_input
    assert "Flipping atom: 0 QMregionindex: 0 Element: O" in caplog.text
    assert theory.brokensym is False
    assert "Flipspin" not in second_input
    assert "*xyz 0 1\n" in second_input


def test_delta_scf_run_writes_the_mom_block_and_then_turns_itself_off(fake_orca, caplog):
    theory = _theory(fake_orca(), delta_scf=True, delta_scf_confline="SetAlpha 4 5 end", delta_scf_pmom=True)

    with caplog.at_level(logging.WARNING, logger="openmmqmmm.orca"):
        _run(theory)

    assert "! DELTASCF \n%scf\n PMOM True \n SetAlpha 4 5 end\nend\n" in Path("orca.inp").read_text()
    assert "Singlet DeltaSCF calculation requested but no UKS/UHF keyword" in caplog.text
    assert theory.delta_scf is False
    assert theory.orcasimpleinput == "! HF def2-SVP nososcf nodamp nolshift"
    assert theory.properties["Mulliken_charges"] == pytest.approx(_mulliken_charges("h2o_engrad"))


def test_ghost_and_dummy_atoms_are_gas_phase_only(fake_orca):
    theory = _theory(fake_orca())
    theory.ghostatoms = [1]
    theory.dummyatoms = [2]

    _run(theory)
    gas_phase_lines = _coordinate_lines()
    _run(theory, pc=True, current_mm_coords=MM_COORDS, mm_charges=MM_CHARGES)
    embedded_lines = _coordinate_lines()

    assert gas_phase_lines == ["O 0.0 0.0 0.1173 ", "H: 0.0 0.7572 -0.4692 ", "DA 0.0 -0.7572 -0.4692 "]
    assert embedded_lines == ["O 0.0 0.0 0.1173 ", "H 0.0 0.7572 -0.4692 ", "H 0.0 -0.7572 -0.4692 "]


def test_first_iteration_input_is_written_only_once(fake_orca):
    theory = _theory(fake_orca(), first_iteration_input="! SlowConv")

    _run(theory)
    first_input = Path("orca.inp").read_text()
    _run(theory)
    second_input = Path("orca.inp").read_text()

    assert first_input.startswith("! HF def2-SVP\n\n! SlowConv\n")
    assert "! SlowConv" not in second_input


def test_xdm_adds_the_dispersion_energy_and_gradient(fake_orca, monkeypatch):
    calls = []

    def fake_xdm(**kwargs):
        calls.append(kwargs)
        return 0.5, np.ones((3, 3))

    monkeypatch.setattr("openmmqmmm.elstructure.xdm_run", fake_xdm)
    theory = _theory(fake_orca(), xdm=True, xdm_a1=0.6, xdm_a2=1.5, xdm_func="PBE")

    energy, gradient = _run(theory, grad=True)

    assert Path("orca.inp").read_text().startswith("! HF def2-SVP AIM\n")
    assert energy == pytest.approx(_final_energy("h2o_engrad") + 0.5)
    assert gradient == pytest.approx(_engrad_gradient("h2o_engrad") + 1.0)
    assert calls == [{"wfxfile": "orca.wfx", "a1": 0.6, "a2": 1.5, "functional": "PBE"}]


def test_run_keeps_labelled_and_numbered_output_copies(fake_orca):
    theory = _theory(fake_orca(), label="water", save_output_with_label=True, keep_each_run_output=True)

    _run(theory)
    _run(theory, charge=1, mult=2)

    output = Path("orca.out").read_text()
    assert Path("orca_water_0_1.out").read_text() == output
    assert Path("orca_water_1_2.out").read_text() == output
    assert Path("orca_run1.out").read_text() == output
    assert Path("orca_run2.out").read_text() == output


def test_tddft_run_collects_the_spectrum(fake_orca):
    text = (FIXTURES / "h2o_tddft.out").read_text()
    energies = [float(v) for v in re.findall(r"^STATE\s+\d+:\s+E=\s+\S+ au\s+(\S+) eV", text, re.MULTILINE)]
    theory = _theory(fake_orca("h2o_tddft"), orcasimpleinput="! PBE def2-SVP def2/J", tddft=True, tddft_roots=3)

    _run(theory)

    assert "%tddft\nnroots 3\nIRoot 1\nend\n" in Path("orca.inp").read_text()
    assert theory.properties["TDDFT_transition_energies"] == pytest.approx(energies)
    assert len(theory.properties["TDDFT_transition_intensities"]) == 3


def test_nmf_run_adds_the_entropy_correction(fake_orca):
    table = (FIXTURES / "h2o_fod.out").read_text().split("ORBITAL ENERGIES", 1)[1].split("MULLIKEN", 1)[0]
    fractions = np.array([float(v) for v in re.findall(r"^\s+\d+\s+(\d\.\d+)\s+-?\d", table, re.MULTILINE)]) / 2
    fractions = fractions[(fractions > 0) & (fractions < 1)]
    entropy_energy = 2 * 0.05 * np.sum(fractions * np.log(fractions) + (1 - fractions) * np.log(1 - fractions))
    theory = _theory(fake_orca("h2o_fod"), orcasimpleinput="! TPSS def2-TZVP", nmf=True, nmf_sigma=0.05)

    energy = _run(theory)

    assert "fracocc true\n" in Path("orca.inp").read_text()
    assert entropy_energy < 0
    assert energy == pytest.approx(_final_energy("h2o_fod") + entropy_energy)


def test_moreadfile_run_reads_orbitals_once(fake_orca):
    Path("start.gbw").write_text("orbitals")
    theory = _theory(fake_orca(), moreadfile="start.gbw")

    _run(theory)

    assert '\n! MOREAD\n%moinp "start.gbw"\n' in Path("orca.inp").read_text()
    assert theory.moreadfile is None


def test_rohf_uhf_swap_run_names_the_second_job_gbw(fake_orca):
    theory = _theory(fake_orca("h2o_cation_charges"), orcasimpleinput="! ROHF def2-SVP", rohf_uhf_swap=True)

    _run(theory, charge=1, mult=2)

    text = Path("orca.inp").read_text()
    assert "\n$new_job\n! UHF noiter  def2-SVP\n\n* xyz 1 2\n" in text
    assert text.count("H 0.0 0.7572 -0.4692 \n") == 2
    assert theory.gbwfile == "orca_job2.gbw"


def test_property_block_follows_the_coordinates(fake_orca):
    theory = _theory(fake_orca("h2o_polar"), propertyblock="%elprop Polar 1 end\n")

    _run(theory)

    assert Path("orca.inp").read_text().endswith("*\n%elprop Polar 1 end\n")
    assert theory.get_polarizability_tensor().shape == (3, 3)


def test_run_prints_population_analysis_on_request(fake_orca, caplog):
    theory = _theory(fake_orca(), print_population_analysis=True)

    with caplog.at_level(logging.INFO, logger="openmmqmmm.orca"):
        _run(theory)

    assert "Printing Mulliken Population analysis" in caplog.text
    assert theory.properties["Mulliken_charges"] == pytest.approx(_mulliken_charges("h2o_engrad"))


def test_orca_launcher_ignores_a_failure_on_request(fake_orca, caplog):
    Path("orca.inp").write_text("! HF def2-SVP\n" + WATER_BLOCK)

    with caplog.at_level(logging.ERROR, logger="openmmqmmm.orca"):
        orca._run_orca_sp_parallel(fake_orca(None, exit_code=1), "orca.inp", ignore_orca_error=True)

    assert "Problem running ORCA" in caplog.text
    assert "error termination in SCF" in caplog.text


def test_external_optimizer_requires_fragment_and_theory():
    with pytest.raises(InputError, match="requires fragment and theory"):
        orca_external_optimizer(fragment=None, theory=ZeroTheory())


def test_external_optimizer_freezes_inactive_atoms_and_accepts_goat(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr("openmmqmmm.orca.find_orca", lambda _orcadir: str(tmp_path))
    monkeypatch.setenv("PATH", os.environ["PATH"])
    monkeypatch.setenv("EXTOPTEXE", "")

    def finish(argv, **kwargs):
        kwargs["stdout"].write("THE OPTIMIZATION HAS CONVERGED\n")
        kwargs["stdout"].write("FINAL SINGLE POINT ENERGY (From external program) -2.0\nTOTAL RUN TIME: 0\n")
        kwargs["stdout"].flush()
        (tmp_path / "ORCAEXTERNAL.xyz").write_text("3\n\nH 0 0 0\nH 0 0 0.8\nH 0 0 3.0\n")

    monkeypatch.setattr("openmmqmmm.orca.sp.run", finish)
    fragment = Fragment(elems=["H", "H", "H"], coords=[[0, 0, 0], [0, 0, 0.74], [0, 0, 3.0]], charge=0, mult=2)

    with caplog.at_level(logging.INFO, logger="openmmqmmm.orca"):
        energy = orca_external_optimizer(fragment=fragment, theory=ZeroTheory(), orca_jobkeyword="GOAT", actatoms=[1])

    text = Path("ORCAEXTERNAL.inp").read_text()
    assert text.startswith("! ExtOpt GOAT\n\n%geom Constraints\n{C 0 C}\n{C 2 C}\nend\nend\n%method\n")
    assert "GOAT keyword found" in caplog.text
    assert energy == -2.0

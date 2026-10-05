import ast
import os

import pytest

from openmmqmmm import Fragment, OpenMMTheory, ORCATheory, QMMMTheory, ZeroTheory, orca_external_optimizer
from openmmqmmm.exceptions import InputError
from openmmqmmm.orca import create_orca_input_pc, create_orca_input_plain

BASE_ARGS = {
    "elems": ["H", "F"],
    "coords": [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
    "orcasimpleinput": "! r2SCAN def2-SVP",
    "orcablockinput": "%scf maxiter 200 end",
    "charge": 0,
    "mult": 1,
}


class OrcaLaunchedError(Exception):
    """Raised in place of launching ORCA, to stop right after input writing."""


def _fail_at_launch(*args, **kwargs):
    raise OrcaLaunchedError


def _directive_lines(path):
    """The `!` keyword and `%block` lines of an input file, in order."""
    return [line for line in path.read_text().splitlines() if line.startswith(("!", "%"))]


def test_keywords_and_blocks_each_get_their_own_line(tmp_path):
    """Every directive must sit on its own line, however the options are combined."""
    for writer in (create_orca_input_plain, create_orca_input_pc):
        name = str(tmp_path / writer.__name__)
        writer(name, extraline="! TightSCF", grad=True, hessian=True, **BASE_ARGS)

        lines = _directive_lines(tmp_path / f"{writer.__name__}.inp")
        assert "! r2SCAN def2-SVP" in lines
        assert "! TightSCF" in lines
        assert "! Engrad" in lines
        assert "! Freq" in lines
        assert "%scf maxiter 200 end" in lines


@pytest.mark.parametrize("extraline", ["! TightSCF", "! TightSCF\n", "\n! Noautostart\n"])
def test_extraline_is_separated_from_what_follows(tmp_path, extraline):
    name = str(tmp_path / "orca")
    create_orca_input_plain(name, extraline=extraline, grad=True, **BASE_ARGS)

    for line in _directive_lines(tmp_path / "orca.inp"):
        assert line.count("!") <= 1, f"Two keyword directives share a line: {line!r}"
        assert not (line.startswith("!") and "%" in line), f"Keyword ran into a block: {line!r}"


def test_pc_and_plain_differ_only_by_the_pointcharge_line(tmp_path):
    """The two writers are one implementation; only `%pointcharges` should differ."""
    create_orca_input_plain(str(tmp_path / "plain"), extraline="! TightSCF", grad=True, **BASE_ARGS)
    create_orca_input_pc(str(tmp_path / "pc"), extraline="! TightSCF", grad=True, **BASE_ARGS)

    plain_lines = (tmp_path / "plain.inp").read_text().splitlines()
    pc_lines = (tmp_path / "pc.inp").read_text().splitlines()

    pointcharge_lines = [line for line in pc_lines if line.startswith("%pointcharges")]
    assert pointcharge_lines == [f'%pointcharges "{tmp_path / "pc"}.pc"']
    assert [line for line in pc_lines if not line.startswith("%pointcharges")] == plain_lines


def test_fragment_indices_keep_unassigned_atoms(tmp_path):
    """Atoms in no fragment (link atoms) are still written, without a fragment tag."""
    args = BASE_ARGS | {"elems": ["H", "F", "H"], "coords": [[0.0, 0.0, 0.0]] * 3}
    create_orca_input_plain(str(tmp_path / "orca"), fragment_indices=[[0, 1]], **args)

    coord_lines = [line for line in (tmp_path / "orca.inp").read_text().splitlines() if line and line[0].isalpha()]
    assert len(coord_lines) == 3, "Every atom must reach the coordinate block"
    assert coord_lines[0].startswith("H(1)")
    assert coord_lines[1].startswith("F(1)")
    assert coord_lines[2].startswith("H "), "The unassigned atom keeps a plain element symbol"


@pytest.mark.usefixtures("fake_orca_dir")
def test_opt_writes_valid_input_and_leaves_theory_unchanged(tmp_path, monkeypatch):
    """Repeated opt() calls must each write valid input and not accumulate state."""
    monkeypatch.setattr("openmmqmmm.orca._run_orca_sp_parallel", _fail_at_launch)

    theory = ORCATheory(orcasimpleinput="! HF def2-SVP", orcablocks="%scf maxiter 200 end")
    fragment = Fragment(coordsstring="H 0.0 0.0 0.0\nF 0.0 0.0 0.95\n", charge=0, mult=1)
    extraline_before = theory.extraline

    for attempt in range(2):
        with pytest.raises(OrcaLaunchedError):
            theory.opt(fragment=fragment)

        assert theory.extraline == extraline_before, f"opt() mutated the theory object (call {attempt + 1})"
        directives = _directive_lines(tmp_path / "orca.inp")
        assert directives.count("! OPT") == 1, f"Expected exactly one OPT directive, got {directives}"
        assert "%scf maxiter 200 end" in directives


def test_generated_otool_script_only_imports_names_that_exist(tmp_path):
    """The otool_external script is written as a string, so no tool can see its imports."""
    import importlib

    from openmmqmmm.orca import write_otool_script

    write_otool_script(basename="mol", theoryfile="theory.pickle", scriptlocation=str(tmp_path), charge=0, mult=1)
    script = (tmp_path / "otool_external").read_text()

    imported = [
        (node.module, alias.name)
        for node in ast.walk(ast.parse(script))
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("openmmqmmm")
        for alias in node.names
    ]
    assert imported, "the generated script no longer imports from openmmqmmm"
    for module_name, name in imported:
        module = importlib.import_module(module_name)
        assert hasattr(module, name), f"otool_external imports {module_name}.{name}, which does not exist"

    # Every name it imports must also be called somewhere in the script it generates
    called = {
        node.func.id
        for node in ast.walk(ast.parse(script))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    for _, name in imported:
        assert name in called, f"otool_external imports {name} but never calls it"


@pytest.fixture
def external_optimizer_stops_at_orca_launch(tmp_path, monkeypatch):
    """Let orca_external_optimizer write its inputs, then stop it where ORCA would be launched."""
    monkeypatch.setenv("PATH", os.environ["PATH"])
    monkeypatch.setenv("EXTOPTEXE", "")
    monkeypatch.setattr("openmmqmmm.orca.find_orca", lambda orcadir=None: str(tmp_path))
    monkeypatch.setattr("openmmqmmm.orca.sp.run", _fail_at_launch)


def _external_optimizer_charge_mult(theory, fragment):
    """The charge/mult baked into the otool_external script and the ORCA input's *xyzfile line."""
    with pytest.raises(OrcaLaunchedError):
        orca_external_optimizer(fragment=fragment, theory=theory)

    with open("otool_external") as script:
        single_point_call = next(
            node
            for node in ast.walk(ast.parse(script.read()))
            if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "single_point"
        )
    script_values = {
        kw.arg: ast.literal_eval(kw.value) for kw in single_point_call.keywords if kw.arg in ("charge", "mult")
    }
    with open("ORCAEXTERNAL.inp") as inp:
        xyzfile_line = next(line for line in inp if line.startswith("*xyzfile"))
    return (script_values["charge"], script_values["mult"]), tuple(xyzfile_line.split()[1:3])


def _two_hydrogen_mm_theory(fragment):
    return OpenMMTheory(
        fragment=fragment,
        dummysystem=True,
        platform="Reference",
        autoconstraints=None,
        rigidwater=False,
        hydrogenmass=None,
    )


def _subregion_qmmm(**kwargs):
    """A QM/MM theory whose QM region (atom 0) is a strict subset of the two-atom system."""
    fragment = Fragment(elems=["H", "H"], coords=[[1.0, 0, 0], [5.0, 0, 0]], charge=0, mult=1, conncalc=False)
    theory = QMMMTheory(
        fragment=fragment,
        qm_theory=ZeroTheory(),
        mm_theory=_two_hydrogen_mm_theory(fragment),
        qmatoms=[0],
        embedding="elstat",
        dipole_correction=False,
        **kwargs,
    )
    return theory, fragment


@pytest.mark.usefixtures("external_optimizer_stops_at_orca_launch")
def test_external_optimizer_bakes_the_qm_region_charge_of_a_qmmm_theory():
    theory, fragment = _subregion_qmmm(qm_charge=-1, qm_mult=1)

    script_charge_mult, xyzfile_charge_mult = _external_optimizer_charge_mult(theory, fragment)

    assert script_charge_mult == (-1, 1)
    assert xyzfile_charge_mult == ("-1", "1")


@pytest.mark.usefixtures("external_optimizer_stops_at_orca_launch")
def test_external_optimizer_rejects_a_subregion_qmmm_theory_without_qm_charge():
    """The fragment's whole-system charge must not stand in for the QM region's."""
    theory, fragment = _subregion_qmmm()

    with pytest.raises(InputError, match="qm_charge"):
        orca_external_optimizer(fragment=fragment, theory=theory)


@pytest.mark.usefixtures("external_optimizer_stops_at_orca_launch")
def test_external_optimizer_takes_a_plain_qm_charge_from_the_fragment():
    fragment = Fragment(elems=["H", "H"], coords=[[0.0, 0, 0], [0.74, 0, 0]], charge=1, mult=2)

    script_charge_mult, xyzfile_charge_mult = _external_optimizer_charge_mult(ZeroTheory(), fragment)

    assert script_charge_mult == (1, 2)
    assert xyzfile_charge_mult == ("1", "2")


@pytest.mark.usefixtures("external_optimizer_stops_at_orca_launch")
def test_external_optimizer_runs_an_mm_theory_on_a_fragment_without_charge():
    """An MM theory has no charge/mult, but ORCA still needs integers on its *xyzfile line."""
    fragment = Fragment(elems=["H", "H"], coords=[[1.0, 0, 0], [5.0, 0, 0]], conncalc=False)

    script_charge_mult, xyzfile_charge_mult = _external_optimizer_charge_mult(
        _two_hydrogen_mm_theory(fragment), fragment
    )

    assert script_charge_mult == (None, None)
    assert xyzfile_charge_mult == ("0", "1")

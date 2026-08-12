"""Tests for ORCA input-file generation.

These functions had no coverage, which is how a missing newline separator shipped:
`create_orca_input_plain` ran `extraline` straight into `orcablocks`, so
`ORCATheory(orcablocks=...).opt()` wrote `! OPT %scf maxiter 200 end` on one line
and ORCA exited with an error. The point-charge variant got the separators right,
and the two writers were 90% duplicated, so the bug only affected gas-phase runs.

Nothing here needs an ORCA installation — these are pure file-writing functions.
"""

import pytest

from openmmqmmm import Fragment, ORCATheory
from openmmqmmm.orca import create_orca_input_pc, create_orca_input_plain

BASE_ARGS = {
    "elems": ["H", "F"],
    "coords": [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
    "orcasimpleinput": "! r2SCAN def2-SVP",
    "orcablockinput": "%scf maxiter 200 end",
    "charge": 0,
    "mult": 1,
}


def _directive_lines(path):
    """The `!` keyword and `%block` lines of an input file, in order."""
    return [line for line in path.read_text().splitlines() if line.startswith(("!", "%"))]


def test_keywords_and_blocks_each_get_their_own_line(tmp_path):
    """Every directive must sit on its own line, however the options are combined.

    ORCA input is line-oriented: two directives sharing a line is a syntax error.
    """
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
    """A user extraline is separated from the next directive whether or not it
    carries its own trailing newline."""
    name = str(tmp_path / "orca")
    create_orca_input_plain(name, extraline=extraline, grad=True, **BASE_ARGS)

    for line in _directive_lines(tmp_path / "orca.inp"):
        assert line.count("!") <= 1, f"Two keyword directives share a line: {line!r}"
        assert not (line.startswith("!") and "%" in line), f"Keyword ran into a block: {line!r}"


def test_pc_and_plain_differ_only_by_the_pointcharge_line(tmp_path):
    """The two writers are one implementation; only `%pointcharges` should differ.

    They used to be separate near-copies that had drifted apart.
    """
    create_orca_input_plain(str(tmp_path / "plain"), extraline="! TightSCF", grad=True, **BASE_ARGS)
    create_orca_input_pc(str(tmp_path / "pc"), extraline="! TightSCF", grad=True, **BASE_ARGS)

    plain_lines = (tmp_path / "plain.inp").read_text().splitlines()
    pc_lines = (tmp_path / "pc.inp").read_text().splitlines()

    pointcharge_lines = [line for line in pc_lines if line.startswith("%pointcharges")]
    assert pointcharge_lines == [f'%pointcharges "{tmp_path / "pc"}.pc"']
    assert [line for line in pc_lines if not line.startswith("%pointcharges")] == plain_lines


def test_fragment_indices_keep_unassigned_atoms(tmp_path):
    """Atoms in no fragment (link atoms) are still written, without a fragment tag.

    The plain writer used to raise TypeError on them and the PC writer used to drop
    them from the coordinate block entirely, silently shrinking the QM region.
    """
    args = BASE_ARGS | {"elems": ["H", "F", "H"], "coords": [[0.0, 0.0, 0.0]] * 3}
    create_orca_input_plain(str(tmp_path / "orca"), fragment_indices=[[0, 1]], **args)

    coord_lines = [line for line in (tmp_path / "orca.inp").read_text().splitlines() if line and line[0].isalpha()]
    assert len(coord_lines) == 3, "Every atom must reach the coordinate block"
    assert coord_lines[0].startswith("H(1)")
    assert coord_lines[1].startswith("F(1)")
    assert coord_lines[2].startswith("H "), "The unassigned atom keeps a plain element symbol"


@pytest.mark.usefixtures("fake_orca_dir")
def test_opt_writes_valid_input_and_leaves_theory_unchanged(tmp_path, monkeypatch):
    """Repeated opt() calls must each write valid input and not accumulate state.

    opt() used to append `! OPT` to self.extraline, so a second call wrote
    `! OPT ! OPT` and ORCA exited with an error; worse, a later run() single point
    silently inherited the `! OPT`. The ORCA launch is stubbed out — the bug was in
    the input written before it.
    """

    class OrcaLaunchedError(Exception):
        """Raised in place of launching ORCA, to stop right after input writing."""

    def fail_at_launch(*args, **kwargs):
        raise OrcaLaunchedError

    monkeypatch.setattr("openmmqmmm.orca.run_orca_sp_parallel", fail_at_launch)

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

import logging

import pytest

from openmmqmmm import configure_logging
from openmmqmmm.utils import (
    basename,
    clean_number,
    column,
    create_conn_dict,
    find_replace_string_in_file,
    insert_line_into_file,
    isint,
    listdiff,
    natural_sort,
    pygrep,
    pygrep2,
    read_intlist_from_file,
    search_list_of_lists_for_index,
    writelisttofile,
    writestringtofile,
)


def test_basename_strips_the_extension_not_the_directory():
    """Unlike os.path.basename this drops the suffix and keeps the path."""
    assert basename("calc.inp") == "calc"
    assert basename("/path/to/calc.out") == "/path/to/calc"


def test_isint():
    assert isint("42") is True
    assert isint("4.2") is False
    assert isint("not a number") is False


def test_listdiff_is_a_sorted_set_difference():
    assert listdiff([1, 2, 3], [1]) == [2, 3]
    assert listdiff([3, 1, 2], [4]) == [1, 2, 3], "Result is sorted"
    assert listdiff([1, 2], [1, 2]) == []


def test_column():
    assert column([[1, 2], [3, 4], [5, 6]], 1) == [2, 4, 6]


def test_natural_sort_orders_numerically():
    """Plain sorting puts frame10 before frame2; natural sorting does not."""
    assert natural_sort(["frame10", "frame2", "frame1"]) == ["frame1", "frame2", "frame10"]


def test_search_list_of_lists_for_index():
    fragments = [[0, 1, 2], [3, 4]]
    assert search_list_of_lists_for_index(4, fragments) == 1
    assert search_list_of_lists_for_index(99, fragments) is None


def test_create_conn_dict_maps_atoms_to_their_molecule():
    """Inverts a list of molecules into atom index -> molecule index."""
    molecule_of = create_conn_dict([[0, 1], [2, 3]])
    assert molecule_of[0] == 0
    assert molecule_of[1] == 0
    assert molecule_of[3] == 1


def test_clean_number_of_a_real_value():
    assert clean_number(complex(1.5, 0.0)) == pytest.approx(1.5)


def test_pygrep_finds_the_line(tmp_path):
    logfile = tmp_path / "calc.out"
    logfile.write_text("first line\nFINAL SINGLE POINT ENERGY   -75.96\nlast line\n")

    assert "-75.96" in pygrep("FINAL SINGLE POINT ENERGY", str(logfile))
    assert pygrep("NOT PRESENT", str(logfile)) is None


def test_pygrep2_returns_every_match(tmp_path):
    logfile = tmp_path / "calc.out"
    logfile.write_text("WARNING: one\nfine\nWARNING: two\n")

    assert len(pygrep2("WARNING", str(logfile))) == 2


def test_insert_line_into_file(tmp_path):
    """This is how the parallel ORCA runner injects its %pal block."""
    inputfile = tmp_path / "orca.inp"
    inputfile.write_text("! HF def2-SVP\n*xyz 0 1\nH 0.0 0.0 0.0\n*\n")

    insert_line_into_file(str(inputfile), "!", "%pal nprocs 4 end", once=True)

    lines = inputfile.read_text().splitlines()
    assert "%pal nprocs 4 end" in lines
    assert lines.index("%pal nprocs 4 end") == 1, "Inserted directly after the matched line"


def test_find_replace_string_in_file(tmp_path):
    target = tmp_path / "input.txt"
    target.write_text("method = HF\nbasis = STO-3G\n")

    find_replace_string_in_file(str(target), "STO-3G", "def2-SVP")

    assert "def2-SVP" in target.read_text()
    assert "STO-3G" not in target.read_text()


def test_write_and_read_intlist(tmp_path):
    """Active-region files are written and read back as plain integer lists."""
    listfile = tmp_path / "actatoms.txt"
    writelisttofile([3, 4, 5], str(listfile))

    assert read_intlist_from_file(str(listfile)) == [3, 4, 5]


def test_read_intlist_applies_the_offset(tmp_path):
    """The offset converts between 1-based file conventions and 0-based indices."""
    listfile = tmp_path / "actatoms.txt"
    writelisttofile([1, 2, 3], str(listfile))

    assert read_intlist_from_file(str(listfile), offset=-1) == [0, 1, 2]


def test_writestringtofile(tmp_path):
    target = tmp_path / "out.txt"
    writestringtofile("hello", str(target))
    assert target.read_text() == "hello"


def test_configure_logging_does_not_stack_handlers():
    """Calling configure_logging repeatedly must not duplicate console output."""
    first = configure_logging()
    handler_count = len(first.handlers)

    second = configure_logging()

    assert second is first
    assert len(second.handlers) == handler_count, "A repeat call replaces its handler rather than adding one"


def test_configure_logging_respects_the_env_override(monkeypatch):
    monkeypatch.setenv("OPENMMQMMM_LOGLEVEL", "WARNING")
    package_logger = configure_logging(level="INFO")
    assert package_logger.level == logging.WARNING


def test_configure_logging_writes_to_a_file(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENMMQMMM_LOGLEVEL", raising=False)
    logfile = tmp_path / "calc.log"

    package_logger = configure_logging(file=str(logfile))
    package_logger.info("a calculation record line")
    for handler in package_logger.handlers:
        handler.flush()

    assert "a calculation record line" in logfile.read_text()

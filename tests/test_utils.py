import io
import logging

import pytest

from openmmqmmm import configure_logging
from openmmqmmm.utils import (
    basename,
    clean_number,
    column,
    find_replace_string_in_file,
    insert_line_into_file,
    isint,
    listdiff,
    natural_sort,
    pygrep,
    pygrep2,
    read_intlist_from_file,
    search_list_of_lists_for_index,
    write_list_to_file,
    write_string_to_file,
)


@pytest.fixture
def isolated_package_logger():
    """Restore package logging after tests that exercise global logger state."""
    package_logger = logging.getLogger("openmmqmmm")
    configured_loggers = (package_logger, logging.getLogger("geometric"))
    original_states = {
        configured_logger: (
            list(configured_logger.handlers),
            configured_logger.level,
            configured_logger.propagate,
            configured_logger.disabled,
        )
        for configured_logger in configured_loggers
    }
    yield package_logger
    handlers_to_close = set()
    for configured_logger, (original_handlers, _level, _propagate, _disabled) in original_states.items():
        for handler in list(configured_logger.handlers):
            configured_logger.removeHandler(handler)
            if handler not in original_handlers:
                handlers_to_close.add(handler)
    for handler in handlers_to_close:
        handler.close()
    for configured_logger, (original_handlers, level, propagate, disabled) in original_states.items():
        for handler in original_handlers:
            configured_logger.addHandler(handler)
        configured_logger.setLevel(level)
        configured_logger.propagate = propagate
        configured_logger.disabled = disabled


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
    write_list_to_file([3, 4, 5], str(listfile))

    assert read_intlist_from_file(str(listfile)) == [3, 4, 5]


def test_read_intlist_applies_the_offset(tmp_path):
    """The offset converts between 1-based file conventions and 0-based indices."""
    listfile = tmp_path / "actatoms.txt"
    write_list_to_file([1, 2, 3], str(listfile))

    assert read_intlist_from_file(str(listfile), offset=-1) == [0, 1, 2]


def test_writestringtofile(tmp_path):
    target = tmp_path / "out.txt"
    write_string_to_file("hello", str(target))
    assert target.read_text() == "hello"


def test_configure_logging_does_not_stack_handlers(capsys, isolated_package_logger):
    """Calling configure_logging repeatedly must not duplicate console output."""
    first = configure_logging()
    second = configure_logging()

    second.info("one calculation record")

    assert second is first
    assert capsys.readouterr().err.count("one calculation record") == 1


def test_configure_logging_does_not_propagate_to_root(isolated_package_logger):
    root_output = io.StringIO()
    root_handler = logging.StreamHandler(root_output)
    root_logger = logging.getLogger()
    root_logger.addHandler(root_handler)
    try:
        package_logger = configure_logging()
        package_logger.info("package-only record")
        logging.getLogger("geometric.test").info("geometric-only record")
    finally:
        root_logger.removeHandler(root_handler)
        root_handler.close()

    assert root_output.getvalue() == ""


def test_configure_logging_includes_geometric_output(capsys, isolated_package_logger):
    configure_logging()

    logging.getLogger("geometric.test").info("optimizer record")

    assert capsys.readouterr().err.count("optimizer record") == 1


def test_configure_logging_respects_the_env_override(monkeypatch, isolated_package_logger):
    monkeypatch.setenv("OPENMMQMMM_LOGLEVEL", "WARNING")
    package_logger = configure_logging(level="INFO")
    assert package_logger.level == logging.WARNING


def test_configure_logging_writes_to_a_file(tmp_path, monkeypatch, isolated_package_logger):
    monkeypatch.delenv("OPENMMQMMM_LOGLEVEL", raising=False)
    logfile = tmp_path / "calc.log"

    package_logger = configure_logging(file=str(logfile))
    package_logger.info("a calculation record line")
    for handler in package_logger.handlers:
        handler.flush()

    assert "a calculation record line" in logfile.read_text()


def test_configure_logging_closes_replaced_file_handlers(tmp_path, isolated_package_logger):
    old_logfile = tmp_path / "old.log"
    package_logger = configure_logging(file=old_logfile)
    old_file_handler = next(handler for handler in package_logger.handlers if isinstance(handler, logging.FileHandler))
    package_logger.info("old record")

    configure_logging()
    package_logger.info("new record")

    assert old_file_handler.stream is None
    assert "new record" not in old_logfile.read_text()


def test_configure_logging_marks_warning_severity(capsys, isolated_package_logger):
    package_logger = configure_logging()

    package_logger.info("ordinary record")
    package_logger.warning("careful record")

    output = capsys.readouterr().err
    assert "ordinary record" in output
    assert "WARNING: careful record" in output


def test_configure_logging_preserves_application_handlers(isolated_package_logger):
    application_output = io.StringIO()
    application_handler = logging.StreamHandler(application_output)
    isolated_package_logger.addHandler(application_handler)

    configure_logging().info("shared record")

    assert application_handler in isolated_package_logger.handlers
    assert application_output.getvalue() == "shared record\n"

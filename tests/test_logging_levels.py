"""Checks that severity is carried by the log level, not by the message text."""

import ast
import pathlib

import pytest

import openmmqmmm

# Located through the import rather than by walking up from this file: the tests live
# outside the package, so they are no longer at a fixed depth below it.
PACKAGE_DIR = pathlib.Path(openmmqmmm.__file__).resolve().parent
# rglob, not glob: the openmm interface is a subpackage and its submodules carry most of
# the logging in the project.
MODULES = sorted(PACKAGE_DIR.rglob("*.py"))

# Bare markers left over from interactive debugging.
DEBUG_MARKERS = {"here", "here1", "here2", "grab true", "test", "ok"}


def _logging_calls(module):
    """Yield (lineno, level, first_argument_string_or_None) for each logger call."""
    tree = ast.parse(module.read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id == "logger"):
            continue
        message = None
        if node.args:
            first = node.args[0]
            if isinstance(first, ast.Constant) and isinstance(first.value, str):
                message = first.value
            elif isinstance(first, ast.JoinedStr):
                message = "".join(v.value for v in first.values if isinstance(v, ast.Constant))
        yield node.lineno, func.attr, message


@pytest.mark.parametrize("module", MODULES, ids=lambda p: p.name)
def test_warnings_are_not_logged_at_info(module):
    offenders = [
        f"{module.name}:{lineno}: {message[:60]!r}"
        for lineno, level, message in _logging_calls(module)
        if level == "info" and message and message.lstrip().lower().startswith(("warning", "error"))
    ]
    assert not offenders, "Warning/error text logged at INFO:\n" + "\n".join(offenders)


@pytest.mark.parametrize("module", MODULES, ids=lambda p: p.name)
def test_no_leftover_debug_markers(module):
    offenders = [
        f"{module.name}:{lineno}: {message!r}"
        for lineno, _level, message in _logging_calls(module)
        if message and message.strip().lower() in DEBUG_MARKERS
    ]
    assert not offenders, "Leftover debug logging:\n" + "\n".join(offenders)


@pytest.mark.parametrize("module", MODULES, ids=lambda p: p.name)
def test_section_banners_are_not_warnings(module):
    """Banner lines belong to the INFO-level calculation record."""
    offenders = [
        f"{module.name}:{lineno}: {message[:60]!r}"
        for lineno, level, message in _logging_calls(module)
        if level in {"warning", "error"} and message and message.lstrip("\n").startswith("---")
    ]
    assert not offenders, "Section banners logged above INFO:\n" + "\n".join(offenders)

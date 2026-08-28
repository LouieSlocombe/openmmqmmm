import re
from pathlib import Path

import pytest

import openmmqmmm
import openmmqmmm.openmm

API_PAGES = sorted((Path(__file__).resolve().parents[1] / "docs" / "source" / "api").glob("*.rst"))

CURRENTMODULE = re.compile(r"^\.\. currentmodule:: (\S+)$")
AUTODOC = re.compile(r"^\.\. auto(?:class|function|exception|data):: (\S+)$")
SUMMARY_ENTRY = re.compile(r"^   (\w+(?:\.\w+)*)$")

# Names documented under openmmqmmm.openmm, because the package root does not export them.
SUBPACKAGE_ONLY = frozenset(openmmqmmm.openmm.__all__) - frozenset(openmmqmmm.__all__)

EXPECTED_DOCUMENTED = {
    *(f"openmmqmmm.{name}" for name in openmmqmmm.__all__),
    *(f"openmmqmmm.openmm.{name}" for name in SUBPACKAGE_ONLY),
}


def _qualify(name, module):
    return name if name.startswith(f"{module}.") else f"{module}.{name}"


def _scan(page):
    """Return the documented and summarised names of one page, module-qualified."""
    documented = []
    summarised = []
    module = "openmmqmmm"
    in_summary = False
    for line in page.read_text().splitlines():
        moduleline = CURRENTMODULE.match(line)
        if moduleline:
            module = moduleline.group(1)
            in_summary = False
            continue
        directive = AUTODOC.match(line)
        if directive:
            documented.append(_qualify(directive.group(1), module))
            in_summary = False
            continue
        if line.startswith(".. autosummary::"):
            in_summary = True
            continue
        if in_summary:
            entry = SUMMARY_ENTRY.match(line)
            if entry:
                summarised.append(_qualify(entry.group(1), module))
            elif line.strip():
                in_summary = False
    return documented, summarised


def test_api_pages_exist():
    assert API_PAGES, "no reference pages found under docs/source/api"


def test_every_public_name_is_documented():
    """Every exported name must appear on a reference page, exactly once."""
    documented = [name for page in API_PAGES for name in _scan(page)[0]]

    duplicates = sorted({name for name in documented if documented.count(name) > 1})
    assert not duplicates, f"documented more than once: {duplicates}"

    missing = sorted(EXPECTED_DOCUMENTED - set(documented))
    assert not missing, f"missing from docs/source/api/: {missing}"

    unexpected = sorted(set(documented) - EXPECTED_DOCUMENTED)
    assert not unexpected, f"documented but not exported: {unexpected}"


TOPIC_PAGES = [page for page in API_PAGES if page.name != "index.rst"]


@pytest.mark.parametrize("page", TOPIC_PAGES, ids=lambda page: page.name)
def test_summary_tables_match_the_page(page):
    """A topic page's autosummary tables must list exactly what the page documents."""
    documented, summarised = _scan(page)
    assert summarised, f"{page.name} has no autosummary table"
    assert sorted(summarised) == sorted(documented)


@pytest.mark.parametrize("page", API_PAGES, ids=lambda page: page.name)
def test_documented_names_are_importable(page):
    """Guards against a typo in a directive, which Sphinx would report only at build time."""
    for name in _scan(page)[0]:
        module, _, attribute = name.rpartition(".")
        namespace = openmmqmmm.openmm if module == "openmmqmmm.openmm" else openmmqmmm
        assert hasattr(namespace, attribute), f"{page.name} documents unknown name {name}"

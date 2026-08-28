# Development

## Running the tests

```bash
pytest
```

from a checkout, which takes about five minutes. The fragment, OpenMM and optimizer tests run
without ORCA, and so do the ORCA input-writing and output-parsing tests — those use a fake
ORCA installation and committed reference output. The four end-to-end QM/MM tests skip
automatically when no ORCA installation is found; set `OPENMMQMMM_ORCADIR` to run them too.

Tests run in isolated temporary directories, so no output files are left behind. The test
data (~2.5 MB) lives in the repository and is not shipped in wheels, so run the suite from a
checkout rather than from an installed copy.

```bash
pip install -e ".[test]"
pytest --cov --cov-report=term
```

## Linting and formatting

```bash
pip install -e ".[dev]"
ruff check .
ruff format --check .
```

The ruff version is pinned identically in `pyproject.toml`, `.pre-commit-config.yaml` and the
CI workflow: the formatter's output shifts between releases, so a skew lets a locally clean
tree fail the CI gate. `pre-commit install` runs both on commit.

## Building the documentation

```bash
pip install -e ".[docs]"
make -C docs strict
```

`strict` is `sphinx-build -W --keep-going`, which is what CI and Read the Docs run
(`fail_on_warning` in `.readthedocs.yaml`), so a warning here is a failed build there. The
result is in `docs/build/html`.

Nothing is mocked in `docs/source/conf.py`: autodoc imports the real package, so the docs
build needs the full runtime stack. The `docs` extra adds only Sphinx, furo, myst-parser and
sphinx-copybutton on top of it.

Two tests guard the reference pages. `tests/test_api_docs.py` requires a docstring on every
exported class, method and function, and freezes the set of public names.
`tests/test_docs.py` requires every one of those names to appear on a reference page, so a new
export cannot silently go undocumented.

## Adding to the public API

1. Add the name to `__all__` in `openmmqmmm/__init__.py`, in the semantic group it belongs to.
2. Add it to `EXPECTED_PUBLIC_EXPORTS` in `tests/test_api_docs.py`.
3. Give it a docstring — one line, and see the note below.
4. Add it to the matching page under `docs/source/api/`.

Steps 2 and 4 are each enforced by a test, so the suite tells you if you miss one.

:::{note}
Docstrings in this package are one-line summaries. The `Args:`/`Returns:`/`Raises:` blocks
were removed deliberately in favour of complete inline type annotations; the API reference is
built on that. Do not add them back.
:::

## Building the package

```bash
python -m build
```

produces an sdist and a wheel under `dist/`. CI installs both and imports them from outside
the source tree, which is what catches a missing package-data entry.

# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

from importlib.metadata import version as _package_version

# -- Project information -----------------------------------------------------

project = "openmmqmmm"
# Sphinx accepts this alias for `copyright`, which would shadow the builtin (ruff A001).
project_copyright = "2026, R. Bjornsson and Louie Slocombe"
author = "Louie Slocombe"

# Single-sourced from the installed package's metadata; pyproject.toml owns the number.
release = _package_version("openmmqmmm")
version = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "myst_parser",
    "sphinx_copybutton",
]

exclude_patterns = []

# Narrative pages are MyST Markdown; the API stubs stay reStructuredText.
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}

# colon_fence keeps ::: admonitions readable when a page is viewed on GitHub;
# deflist is used for the option tables in the guides.
myst_enable_extensions = ["colon_fence", "deflist"]
myst_heading_anchors = 3

# -- Intersphinx -------------------------------------------------------------
# Only projects that actually publish an objects.inv are listed. ASE and mdtraj
# do not, so their types render as plain text rather than links.
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "openmm": ("http://docs.openmm.org/latest/api-python", None),
    "parmed": ("https://parmed.github.io/ParmEd/html", None),
}
# A docs site being unreachable must not fail a release build.
intersphinx_timeout = 30

# -- Cross-reference strictness ----------------------------------------------
# Nitpicky mode is off, deliberately. OpenMM's inventory registers its classes
# under their defining modules (openmm.openmm.System, openmm.app.topology.Topology)
# while this package annotates them by the import-path aliases OpenMM itself
# documents (openmm.System, openmm.app.Topology). Nitpicky mode would report every
# such annotation as an unresolved reference and drown the fail_on_warning gate.
# Prose that wants a live link writes the inventory path: :class:`~openmm.openmm.System`.
nitpicky = False

# -- Autodoc -----------------------------------------------------------------
# Nothing is mocked: autodoc imports the real package, so a missing runtime
# dependency surfaces as a failed build rather than an empty page. Install with
# `pip install -e ".[docs]"` from the repository root.
autodoc_mock_imports = []

autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
    "member-order": "bysource",
}

# The 2026-08 prose audit removed every Args:/Returns: block, so the inline
# annotation is the parameter documentation and has to stay in the signature.
autodoc_typehints = "signature"
autodoc_preserve_defaults = True

# openmm_md, openmm_modeller and OpenMMTheory take dozens of keyword arguments;
# one parameter per line is the only readable rendering of those signatures.
python_maximum_signature_line_length = 88

# The API pages name every object explicitly, so stub generation would only
# duplicate them.
autosummary_generate = False

# -- Napoleon ----------------------------------------------------------------
# Google style, matching [tool.ruff.lint.pydocstyle] in pyproject.toml. NumPy
# parsing is off so a malformed docstring surfaces instead of half-rendering.
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = False
napoleon_use_ivar = True
napoleon_use_param = True
napoleon_use_rtype = True

# -- HTML output -------------------------------------------------------------

html_theme = "furo"
html_static_path = ["_static"]
html_title = f"openmmqmmm {version}"

html_theme_options = {
    "source_repository": "https://github.com/LouieSlocombe/openmmqmmm/",
    "source_branch": "main",
    "source_directory": "docs/source/",
}

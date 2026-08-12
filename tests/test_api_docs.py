"""Checks that the public API stays documented and that py.typed is honest.

The 1.0.0 docstring pass covered module-level functions only, so all 128 public
methods of the exported classes — everything a user calls on a Fragment, an
ORCATheory or an OpenMMTheory — went undocumented while the README advertised a
documented API. These tests fail the moment a new public method arrives without one.
"""

import inspect
import pathlib

import pytest

import openmmqmmm

EXPORTED_CLASSES = [
    name
    for name in openmmqmmm.__all__
    if inspect.isclass(getattr(openmmqmmm, name, None))
    and getattr(getattr(openmmqmmm, name), "__module__", "").startswith("openmmqmmm")
]


def _own_public_methods(cls):
    return [
        (name, func)
        for name, func in inspect.getmembers(cls, inspect.isfunction)
        if not name.startswith("_") and func.__module__.startswith("openmmqmmm")
    ]


@pytest.mark.parametrize("class_name", EXPORTED_CLASSES)
def test_exported_classes_are_documented(class_name):
    cls = getattr(openmmqmmm, class_name)
    assert inspect.getdoc(cls), f"{class_name} has no class docstring"


@pytest.mark.parametrize("class_name", EXPORTED_CLASSES)
def test_public_methods_are_documented(class_name):
    cls = getattr(openmmqmmm, class_name)
    undocumented = [name for name, func in _own_public_methods(cls) if not inspect.getdoc(func)]
    assert not undocumented, f"{class_name} has undocumented public methods: {undocumented}"


@pytest.mark.parametrize(
    "name", [name for name in openmmqmmm.__all__ if inspect.isfunction(getattr(openmmqmmm, name, None))]
)
def test_exported_functions_are_documented(name):
    assert inspect.getdoc(getattr(openmmqmmm, name)), f"{name} has no docstring"


def test_py_typed_marker_matches_reality():
    """Shipping py.typed tells type checkers the package is annotated.

    If the marker is present, the exported job functions must actually carry
    annotations — otherwise downstream mypy users are promised types and silently
    get Any. Delete the marker rather than let it lie.
    """
    package_dir = pathlib.Path(openmmqmmm.__file__).parent
    if not (package_dir / "py.typed").exists():
        pytest.skip("No py.typed marker: the package does not claim to be typed")

    unannotated = []
    for name in openmmqmmm.__all__:
        obj = getattr(openmmqmmm, name, None)
        if not inspect.isfunction(obj) or not obj.__module__.startswith("openmmqmmm"):
            continue
        signature = inspect.signature(obj)
        if signature.return_annotation is inspect.Signature.empty:
            unannotated.append(name)

    assert not unannotated, (
        "py.typed is shipped but these exported functions have no return annotation: "
        f"{sorted(unannotated)}. Annotate them or remove the py.typed marker."
    )

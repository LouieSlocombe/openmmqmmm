import inspect

import pytest

import openmmqmmm
import openmmqmmm.openmm

PUBLIC_NAMESPACES = (openmmqmmm, openmmqmmm.openmm)


def _exported_objects(predicate):
    return [
        pytest.param(namespace, name, id=f"{namespace.__name__}.{name}")
        for namespace in PUBLIC_NAMESPACES
        for name in namespace.__all__
        if predicate(getattr(namespace, name, None))
        and getattr(getattr(namespace, name), "__module__", "").startswith("openmmqmmm")
    ]


EXPORTED_CLASSES = _exported_objects(inspect.isclass)
EXPORTED_FUNCTIONS = _exported_objects(inspect.isfunction)


def _own_public_methods(cls):
    return [
        (name, func)
        for name, func in inspect.getmembers(cls, inspect.isfunction)
        if not name.startswith("_") and func.__module__.startswith("openmmqmmm")
    ]


@pytest.mark.parametrize(("namespace", "class_name"), EXPORTED_CLASSES)
def test_exported_classes_are_documented(namespace, class_name):
    cls = getattr(namespace, class_name)
    assert inspect.getdoc(cls), f"{class_name} has no class docstring"


@pytest.mark.parametrize(("namespace", "class_name"), EXPORTED_CLASSES)
def test_public_methods_are_documented(namespace, class_name):
    cls = getattr(namespace, class_name)
    undocumented = [name for name, func in _own_public_methods(cls) if not inspect.getdoc(func)]
    assert not undocumented, f"{class_name} has undocumented public methods: {undocumented}"


@pytest.mark.parametrize(("namespace", "name"), EXPORTED_FUNCTIONS)
def test_exported_functions_are_documented(namespace, name):
    assert inspect.getdoc(getattr(namespace, name)), f"{name} has no docstring"


@pytest.mark.parametrize("namespace", PUBLIC_NAMESPACES, ids=lambda namespace: namespace.__name__)
def test_explicit_exports_are_unique_and_defined(namespace):
    """The export set is pinned by tests/test_docs.py, which matches it against the reference pages."""
    assert len(namespace.__all__) == len(set(namespace.__all__))
    missing = [name for name in namespace.__all__ if not hasattr(namespace, name)]
    assert not missing, f"{namespace.__name__} exports undefined names: {missing}"

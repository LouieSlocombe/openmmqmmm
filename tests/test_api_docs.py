import inspect

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

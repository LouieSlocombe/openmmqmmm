"""Static check that internal keyword arguments match the signatures they are passed to.

The package renames aggressively (the v1.0.0 PEP8 sweep, then a follow-up audit that
restored OpenMM's own camelCase spellings at the call sites that talk to OpenMM). Both
sweeps hit call sites the tests do not exercise: all four OpenMM MD entry points spent a
release passing ``enforcePeriodicBox=`` to ``MolecularDynamicsEngine``, whose parameter is
``enforce_periodic_box``. Every call raised TypeError; openmm/md.py sits at 4% coverage,
so nothing noticed.

Parsing the call sites and comparing them against the real signatures catches that whole
family at import speed, with no OpenMM or ORCA run needed.
"""

import ast
import importlib
import inspect
import pathlib

import pytest

import openmmqmmm

PACKAGE_DIR = pathlib.Path(openmmqmmm.__file__).parent
SOURCE_FILES = sorted(PACKAGE_DIR.rglob("*.py"))


def _public_callables():
    """Map name -> callable for everything the package defines, where the name is unique.

    Names bound to more than one object are dropped: ``write_xyzfile`` is both a module
    function and a Fragment method, and this check has no type information to tell them
    apart.
    """
    found = {}
    for path in SOURCE_FILES:
        module_name = str(path.relative_to(PACKAGE_DIR.parent).with_suffix("")).replace("/", ".")
        module_name = module_name.removesuffix(".__init__")
        module = importlib.import_module(module_name)
        for attr, obj in vars(module).items():
            if inspect.isfunction(obj) and getattr(obj, "__module__", "").startswith("openmmqmmm"):
                found.setdefault(attr, set()).add(obj)
            elif inspect.isclass(obj) and getattr(obj, "__module__", "").startswith("openmmqmmm"):
                found.setdefault(attr, set()).add(obj.__init__)
    return {name: next(iter(objs)) for name, objs in found.items() if len(objs) == 1}


def _accepted_parameters(func):
    """Parameter names func accepts, or None if it takes **kwargs and accepts anything."""
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return None
    if any(p.kind is p.VAR_KEYWORD for p in signature.parameters.values()):
        return None
    return set(signature.parameters)


def _bad_call_sites():
    callables = _public_callables()
    problems = []
    for path in SOURCE_FILES:
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            # Bare-name calls only. `fragment.write_xyzfile(...)` names the Fragment
            # method, not the module function that shares its name, and nothing here can
            # resolve the receiver's type.
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                continue
            target = callables.get(node.func.id)
            if target is None:
                continue
            accepted = _accepted_parameters(target)
            if accepted is None:
                continue
            given = {kw.arg for kw in node.keywords if kw.arg}
            unknown = given - accepted
            if unknown:
                relative = path.relative_to(PACKAGE_DIR.parent)
                problems.append(f"{relative}:{node.lineno} {node.func.id}(...) does not accept {sorted(unknown)}")
    return problems


def test_no_call_site_passes_an_unknown_keyword():
    problems = _bad_call_sites()
    assert not problems, "Keyword arguments that no signature accepts:\n" + "\n".join(problems)


@pytest.mark.parametrize(
    "entry_point",
    ["openmm_md", "openmm_box_equilibration", "openmm_metadynamics", "openmm_md_plumed"],
)
def test_md_entry_points_forward_every_argument_they_accept(entry_point):
    """The MD wrappers restate the engine's parameters; none may be silently dropped.

    Each wrapper takes ~45 keyword arguments and hands them to MolecularDynamicsEngine,
    or to its run() method. An argument accepted by the wrapper but passed to neither is
    accepted from the user and then ignored, which is worse than rejecting it.
    """
    from openmmqmmm.openmm.md import MolecularDynamicsEngine

    module = importlib.import_module(getattr(openmmqmmm, entry_point).__module__)
    function = getattr(module, entry_point)

    engine_parameters = set(inspect.signature(MolecularDynamicsEngine.__init__).parameters)
    wrapper_parameters = set(inspect.signature(function).parameters)

    source = ast.parse(inspect.getsource(function))
    forwarded = set()
    for node in ast.walk(source):
        if not isinstance(node, ast.Call):
            continue
        forwarded |= {kw.arg for kw in node.keywords if kw.arg}
        # **kwargs forwarding to the engine covers everything the wrapper accepts
        if getattr(node.func, "id", None) == "MolecularDynamicsEngine" and any(kw.arg is None for kw in node.keywords):
            forwarded |= wrapper_parameters & engine_parameters

    dropped = sorted((wrapper_parameters & engine_parameters) - forwarded)
    assert not dropped, f"{entry_point} accepts but never forwards: {dropped}"

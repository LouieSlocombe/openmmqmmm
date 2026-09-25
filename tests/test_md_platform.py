"""Platform requests must agree with the MM theory whose System MD reuses."""

import openmm
import pytest
from conftest import _AnalyticQM

from openmmqmmm import Fragment, MolecularDynamicsEngine, OpenMMTheory, QMMMTheory, openmm_md
from openmmqmmm.exceptions import InputError


def _make_theory(kind, platform="Reference", **mm_options):
    fragment = Fragment(elems=["H", "H"], coords=[[-0.5, 0, 0], [0.5, 0, 0]], charge=0, mult=1)
    mm = OpenMMTheory(
        fragment=fragment,
        dummysystem=True,
        platform=platform,
        autoconstraints=None,
        rigidwater=False,
        hydrogenmass=None,
        **mm_options,
    )
    theory = mm
    if kind == "qmmm":
        theory = QMMMTheory(
            fragment=fragment,
            qm_theory=_AnalyticQM(),
            mm_theory=mm,
            qmatoms=[0, 1],
            embedding="mech",
            qm_charge=0,
            qm_mult=1,
        )
    return fragment, theory, mm


def _run_md(entry_point, fragment, theory, **platform_options):
    options = dict(fragment=fragment, theory=theory, timestep=0.000001, **platform_options)
    if entry_point == "wrapper":
        openmm_md(simulation_steps=1, **options)
    else:
        engine = MolecularDynamicsEngine(**options)
        try:
            engine.run(simulation_steps=1)
        finally:
            engine.close()


@pytest.fixture
def context_platforms(monkeypatch):
    """Record the actual platform chosen by each real Context, including wrapper runs."""
    platforms = []
    create_simulation = OpenMMTheory.create_simulation

    def record_platform(theory, internal=False):
        result = create_simulation(theory, internal=internal)
        simulation = theory.simulation if internal else result
        context = simulation.context
        platform = context.getPlatform()
        properties = {name: platform.getPropertyValue(context, name) for name in platform.getPropertyNames()}
        platforms.append((platform.getName(), properties))
        return result

    monkeypatch.setattr(OpenMMTheory, "create_simulation", record_platform)
    return platforms


@pytest.mark.parametrize("kind", ["mm", "qmmm"])
@pytest.mark.parametrize("entry_point", ["engine", "wrapper"])
@pytest.mark.parametrize("platform", ["Reference", "CPU"])
@pytest.mark.parametrize("platform_request", ["omitted", "none", "matching"])
def test_md_inherits_or_accepts_matching_platform(kind, entry_point, platform, platform_request, context_platforms):
    fragment, theory, mm = _make_theory(kind, platform, numcores=2)
    properties = mm.properties.copy()
    options = {} if platform_request == "omitted" else {"platform": None if platform_request == "none" else platform}

    _run_md(entry_point, fragment, theory, **options)

    assert context_platforms
    assert all(name == platform for name, _ in context_platforms)
    if platform == "CPU":
        assert all(props["Threads"] == "2" for _, props in context_platforms)
    assert mm.platform_choice == platform
    assert mm.properties == properties
    assert mm.numcores == 2


@pytest.mark.parametrize("kind", ["mm", "qmmm"])
@pytest.mark.parametrize("entry_point", ["engine", "wrapper"])
@pytest.mark.parametrize("platform", ["CPU", "CUDA"])
def test_md_rejects_conflicting_platform_before_mutating_theory(kind, entry_point, platform, monkeypatch):
    fragment, theory, mm = _make_theory(kind)
    system_xml = openmm.XmlSerializer.serialize(mm.system)
    properties = mm.properties.copy()

    def unexpected_context(*args, **kwargs):
        pytest.fail("A conflicting platform must be rejected before Context creation")

    monkeypatch.setattr(OpenMMTheory, "create_simulation", unexpected_context)
    with pytest.raises(InputError, match=f"Requested MD platform '{platform}'.*platform 'Reference'") as exc:
        _run_md(entry_point, fragment, theory, platform=platform)

    message = str(exc.value)
    assert "Omit platform" in message
    assert f"OpenMMTheory with platform='{platform}'" in message
    if kind == "qmmm":
        assert "QMMMTheory.mm_theory" in message
    assert mm.platform_choice == "Reference"
    assert mm.properties == properties
    assert openmm.XmlSerializer.serialize(mm.system) == system_xml


@pytest.mark.parametrize("entry_point", ["engine", "wrapper"])
@pytest.mark.parametrize(
    "options, expected", [({}, "CPU"), ({"platform": None}, "CPU"), ({"platform": "Reference"}, "Reference")]
)
def test_standalone_qm_platform_default_and_explicit_request(entry_point, options, expected, context_platforms):
    fragment = Fragment(elems=["H", "H"], coords=[[-0.5, 0, 0], [0.5, 0, 0]], charge=0, mult=1)

    _run_md(entry_point, fragment, _AnalyticQM(), **options)

    assert context_platforms
    assert all(name == expected for name, _ in context_platforms)

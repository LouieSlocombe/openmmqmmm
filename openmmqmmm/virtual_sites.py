"""Native OpenMM virtual-site placement and standalone gradient redistribution."""

from __future__ import annotations

from typing import Any

import numpy as np
import openmm
from openmm import unit

from openmmqmmm.exceptions import InputError


def _copy_site(site: openmm.VirtualSite) -> openmm.VirtualSite:
    """Copy a native site without serializing an owning force or QM callback."""
    particles = [site.getParticle(i) for i in range(site.getNumParticles())]
    if isinstance(site, openmm.TwoParticleAverageSite):
        return openmm.TwoParticleAverageSite(*particles, *(site.getWeight(i) for i in range(2)))
    if isinstance(site, openmm.ThreeParticleAverageSite):
        return openmm.ThreeParticleAverageSite(*particles, *(site.getWeight(i) for i in range(3)))
    if isinstance(site, openmm.OutOfPlaneSite):
        return openmm.OutOfPlaneSite(*particles, site.getWeight12(), site.getWeight13(), site.getWeightCross())
    if isinstance(site, openmm.LocalCoordinatesSite):
        return openmm.LocalCoordinatesSite(
            particles, site.getOriginWeights(), site.getXWeights(), site.getYWeights(), site.getLocalPosition()
        )
    raise InputError(f"Unsupported OpenMM virtual site: {type(site).__name__}")


class NativeVirtualSites:
    """Use OpenMM's own placement rules and Jacobians in standalone QM/MM calls.

    The auxiliary Reference context contains only native sites and a linear force
    on those sites. It never evaluates the live MM system or its PythonForce.
    Only serialized system metadata survives copying/pickling a QM/MM theory.
    """

    def __init__(self, system: openmm.System) -> None:
        skeleton = openmm.System()
        self.indices = [i for i in range(system.getNumParticles()) if system.isVirtualSite(i)]
        for i in range(system.getNumParticles()):
            skeleton.addParticle(0 if system.isVirtualSite(i) else 1)
        self.parents = {}
        for i in self.indices:
            site = system.getVirtualSite(i)
            skeleton.setVirtualSite(i, _copy_site(site))
            self.parents[i] = site.getParticle(0)
        force = openmm.CustomExternalForce("gx*x+gy*y+gz*z")
        for name in ("gx", "gy", "gz"):
            force.addPerParticleParameter(name)
        for i in self.indices:
            force.addParticle(i, [0, 0, 0])
        skeleton.addForce(force)
        self._system_xml = openmm.XmlSerializer.serialize(skeleton)
        self._system = None
        self._context = None
        self._force = None
        self._integrator = None

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        for key in ("_system", "_context", "_force", "_integrator"):
            state[key] = None
        return state

    def _get_context(self) -> openmm.Context:
        if self._context is None:
            self._system = openmm.XmlSerializer.deserialize(self._system_xml)
            self._force = self._system.getForce(0)
            self._integrator = openmm.VerletIntegrator(0.001)
            self._context = openmm.Context(
                self._system, self._integrator, openmm.Platform.getPlatformByName("Reference")
            )
        return self._context

    def seed(self, coords: np.ndarray) -> np.ndarray:
        """Ignore supplied site rows while unwrapping the real host molecules."""
        seeded = np.asarray(coords, dtype=float).copy()
        for i in self.indices:
            parent = self.parents[i]
            visited = {i}
            while parent in self.parents:
                if parent in visited:
                    raise InputError("OpenMM virtual sites have cyclic parent dependencies")
                visited.add(parent)
                parent = self.parents[parent]
            seeded[i] = seeded[parent]
        return seeded

    def place(self, coords: np.ndarray) -> np.ndarray:
        """Recompute site coordinates in Å from contiguous host coordinates."""
        context = self._get_context()
        context.setPositions(np.asarray(coords) * 0.1)
        context.computeVirtualSites()
        return np.asarray(context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.angstrom))

    def project(self, coords: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        """Redistribute site gradients to independent hosts, leaving site rows zero."""
        context = self._get_context()
        context.setPositions(np.asarray(coords) * 0.1)
        context.computeVirtualSites()
        for row, index in enumerate(self.indices):
            self._force.setParticleParameters(row, index, np.asarray(gradient[index]).tolist())
        self._force.updateParametersInContext(context)
        # The linear force uses the input numbers as coefficients. Its Jacobian
        # redistributes those same numbers, regardless of their physical units.
        redistributed = -np.asarray(
            context.getState(getForces=True)
            .getForces(asNumpy=True)
            .value_in_unit(unit.kilojoule_per_mole / unit.nanometer)
        )
        projected = np.asarray(gradient).copy()
        projected[self.indices] = 0
        projected += redistributed
        # OpenMM retains raw site forces as well as adding them to the hosts.
        projected[self.indices] = 0
        return projected

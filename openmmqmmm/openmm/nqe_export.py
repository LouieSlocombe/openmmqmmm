"""Export a QM/MM potential as plain OpenMM objects for external NQE drivers.

openmmqmmm and external NQE workflow packages (openmmnqe being the motivating
one) deliberately share no imports: they meet at plain OpenMM types. The
export is the MM System with the bead-specific QM/MM ``openmm.PythonForce``
already attached, plus an ``openmm.app.Modeller`` carrying the topology and
current coordinates, so any driver that steps an OpenMM System can run the
QM/MM potential.

Exporting configures the theory for external-force MD, the same state
``MolecularDynamicsEngine`` leaves it in: standalone ``single_point`` or
geometry optimization on the same theory object would skip the MM part
afterwards, so build a fresh theory for those. Build a fresh export per
driver stage that mutates the System (a barostat, PLUMED bias, or
deuteration), pass ``barostat_freq=None`` to openmmnqe RPMD production
stages, and do not run the exported System through openmmnqe's
``run_openmm_rpmd_contracted``: its force-group reassignment folds the
PythonForce into a contracted group and silently averages the QM force.
"""

import dataclasses
import logging
from numbers import Integral

import numpy as np
import openmm
import openmm.app
import openmm.unit

from openmmqmmm.coords import check_charge_mult
from openmmqmmm.exceptions import InputError
from openmmqmmm.openmm.rpmd_force import RPMDQMMMForceProvider, add_rpmd_python_force
from openmmqmmm.qmmm import QMMMTheory

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class RPMDPotentialExport:
    """Plain OpenMM handles for a QM/MM potential exported to an external driver."""

    system: openmm.System
    modeller: openmm.app.Modeller
    provider: RPMDQMMMForceProvider
    python_force: "openmm.PythonForce"
    force_group: int
    num_beads: int


def modeller_from_topology(*, topology, coords_angstrom) -> openmm.app.Modeller:
    """Build an ``openmm.app.Modeller`` from an OpenMM topology and coordinates in Å."""
    coords = np.asarray(coords_angstrom, dtype=np.float64)
    num_atoms = topology.getNumAtoms()
    if coords.shape != (num_atoms, 3):
        raise InputError(
            f"coords_angstrom has shape {coords.shape} but the topology has {num_atoms} atoms; "
            "expected an (n_atoms, 3) array matching the topology."
        )
    if not np.all(np.isfinite(coords)):
        raise InputError("coords_angstrom must be finite.")
    positions = openmm.unit.Quantity(coords * 0.1, openmm.unit.nanometer)  # Å -> nm
    return openmm.app.Modeller(topology, positions)


def attach_qmmm_rpmd_force(*, theory, elems, charge, mult, num_beads, periodic, cache_size=None):
    """Attach the bead-specific QM/MM ``PythonForce`` to the theory's MM System.

    Shared wiring between ``MolecularDynamicsEngine`` and ``export_rpmd_potential``:
    validates RPMD-incompatible theory options, switches the theory into
    external-force mode, and returns ``(provider, python_force, force_group)``.
    """
    if theory.truncated_pc:
        raise InputError(
            "QM/MM RPMD does not support truncated_pc because its correction history is shared across "
            "beads. Disable truncated_pc for bead-resolved dynamics."
        )
    if theory.update_qm_region_charges:
        raise InputError(
            "QM/MM RPMD does not support update_qm_region_charges because one shared MM charge set "
            "cannot represent every bead."
        )
    if isinstance(num_beads, bool) or not isinstance(num_beads, Integral) or num_beads < 1:
        raise InputError("num_beads must be a positive integer matching the RPMD copy count the System will run under.")
    num_beads = int(num_beads)

    # The provider evaluates only the QM and coupling terms; these flags make
    # QMMMTheory.run skip the MM part, which the System's native forces own.
    theory.exit_after_customexternalforce_update = True
    theory.openmm_externalforce = True

    if cache_size is None:
        # RPMDIntegrator evaluates the potential twice per step at every bead.
        cache_size = 2 * num_beads + 4

    provider = RPMDQMMMForceProvider(
        theory,
        elems,
        charge,
        mult,
        periodic=periodic,
        cache_size=cache_size,
    )
    python_force, force_group = add_rpmd_python_force(theory.mm_theory.system, provider, periodic=periodic)
    return provider, python_force, force_group


def export_rpmd_potential(
    *, theory, num_beads, fragment=None, charge=None, mult=None, cache_size=None
) -> RPMDPotentialExport:
    """Export a QM/MM theory as a System, Modeller, and attached ``PythonForce``.

    The returned System is ``theory.mm_theory.system`` itself, carrying the
    QM/MM ``PythonForce`` in its own force group; the Modeller pairs the MM
    topology with the fragment coordinates. ``num_beads`` must match the RPMD
    copy count the external driver will run (it sizes the provider's
    coordinate cache); use ``num_beads=1`` for classical or adQTB drivers.
    """
    if not isinstance(theory, QMMMTheory):
        raise InputError(
            f"export_rpmd_potential requires a QMMMTheory, got {type(theory).__name__}. "
            "Wrap the QM and MM sides in a QMMMTheory first."
        )
    if fragment is None:
        fragment = theory.fragment

    charge, mult = check_charge_mult(charge, mult, theory.theorytype, fragment, "export_rpmd_potential", theory=theory)

    # Build the Modeller first: its shape validation runs before anything mutates
    # the theory or its System.
    modeller = modeller_from_topology(topology=theory.mm_theory.topology, coords_angstrom=fragment.coords)
    provider, python_force, force_group = attach_qmmm_rpmd_force(
        theory=theory,
        elems=fragment.elems,
        charge=charge,
        mult=mult,
        num_beads=num_beads,
        periodic=theory.mm_theory.periodic,
        cache_size=cache_size,
    )
    logger.info("Exported QM/MM potential for %s RPMD beads in force group %s", num_beads, force_group)
    return RPMDPotentialExport(
        system=theory.mm_theory.system,
        modeller=modeller,
        provider=provider,
        python_force=python_force,
        force_group=force_group,
        num_beads=int(num_beads),
    )

# ML/MM

{class}`~openmmqmmm.MLMMTheory` replaces the internal force-field interactions of a
selected region with an [OpenMM-ML](https://github.com/openmm/openmm-ml) potential.
The surrounding system and interactions across the ML/MM boundary retain the classical
force field. This is mechanical embedding: MM charges do not polarize the ML model.

Install the optional {ref}`OpenMM-ML dependencies <install-openmm-ml>` and the chosen
model's backend and weights first. ORCA is not needed.

## Build a mixed theory

Start with a fully parameterized {class}`~openmmqmmm.OpenMMTheory`, including classical
parameters for the ML atoms. The `Fragment`, topology and system must use the same atom
order. `mlatoms` contains zero-based topology atom indices and must select whole molecules:
no covalent bond may cross the boundary, and ML atoms cannot be virtual sites.

```python
from openmmqmmm import Fragment, MLMMTheory, OpenMMTheory, openmm_md, single_point

fragment = Fragment(pdbfile="system.pdb")
mm = OpenMMTheory(
    xmlfiles=["amber14-all.xml", "amber14/tip3pfb.xml", "ligand.xml"],
    pdbfile="system.pdb",
    periodic=True,
    autoconstraints=None,
    rigidwater=False,
    hydrogenmass=None,
    platform="CPU",
)

# Select a complete ligand molecule; residue names need not be unique.
mlatoms = [atom.index for atom in mm.topology.atoms() if atom.residue.name == "LIG"]
mlmm = MLMMTheory(mm_theory=mm, mlatoms=mlatoms, potential="ani2x")

result = single_point(theory=mlmm, fragment=fragment)
print(f"ML/MM energy: {result.energy:.8f} Eh")
openmm_md(fragment=fragment, theory=mlmm, timestep=0.0005, simulation_time=1)
```

The ligand XML must describe the input ligand, and the selected ML model must support its
chemistry. `MLMMTheory` makes an independent copy of the source MM system and topology;
constructing it leaves `mm` available for classical calculations. An empty selection,
duplicate or out-of-range indices, a cut covalent bond or an ML virtual site raises
{exc}`~openmmqmmm.InputError`.

Use a classical source theory; an `MLMMTheory` or a source already modified by
`QMMMTheory` is rejected.
Standard bonded and nonbonded forces are supported; implicit-solvent, polarizable and
unsupported many-body force types are rejected to prevent double counting. Nonbonded
parameter offsets that alter removed ML interactions are also rejected. Restraints created
by `OpenMMTheory` are retained.
All topology atoms must have elements: OpenMM-ML 1.8 currently prevents using extra
particles without elements, including those in common four-site water models.

`potential` also accepts an existing `openmmml.MLPotential` object. With a name,
`potential_kwargs` supplies its constructor arguments. `system_kwargs` supplies
model-specific arguments to `createMixedSystem`, for example
`system_kwargs={"device": "cuda"}` for a compatible backend. Set the OpenMM platform and
its properties on the source MM theory; the ML backend's device is a separate setting.
Consult the selected model's documentation for both options and dependencies.

The wrapper fixes mechanical embedding without interpolation. `system_kwargs` cannot
override `topology`, `system`, `atoms`, `embedding`, `interpolate`, `returnInfo` or
`removeConstraints`; use the explicit `remove_constraints` argument for the last option.
`potential_kwargs` configures named potentials only.

## MACE-POLAR

`potential="mace-polar"` selects `mace-polar-1-medium`. The explicit names
`mace-polar-1-small`, `mace-polar-1-medium` and `mace-polar-1-large` select the corresponding
[OpenMM-ML models](https://github.com/openmm/openmm-ml/blob/1.8/openmmml/models/macepotential.py).
Install the {ref}`MACE-POLAR backend <install-mace-polar>` before constructing the theory.

```python
polar_mlmm = MLMMTheory(
    mm_theory=mm,
    mlatoms=mlatoms,
    potential="mace-polar",
    system_kwargs={"charge": -1, "multiplicity": 1, "device": "cuda"},
)
```

`charge` and `multiplicity` describe the entire selected ML subset, not the full solvated
system. Their defaults are 0 and 1; set them explicitly for a different electronic state.
Job-level `charge` and `mult` do not reconfigure the model. Rebuild `MLMMTheory` to change
its state. `precision` defaults to `"double"`; the backend device remains independent of
the source theory's OpenMM platform.

The wrapper requires `returnEnergyType="energy"` for MACE-POLAR. The upstream
`interaction_energy` output omits electrostatic contributions, whereas its forces derive
from the full energy; using both together would give inconsistent energies and forces.
See the [PolarMACE implementation](https://github.com/ACEsuit/mace/blob/v0.3.16/mace/modules/extensions.py).

For a local MACE-POLAR checkpoint, retain the polar alias and supply its path:

```python
polar_mlmm = MLMMTheory(
    mm_theory=mm,
    mlatoms=mlatoms,
    potential="mace-polar",
    potential_kwargs={"modelPath": "/path/to/MACE-POLAR.model"},
    system_kwargs={"charge": -1, "multiplicity": 1, "device": "cuda"},
)
```

Local checkpoints use the same electronic-state settings, long-range treatment and
full-energy safeguard. Named foundation models may download weights on first use; a local
checkpoint avoids that download. For periodic systems, use a source MM theory with PME or
Ewald; LJPME is unsupported.

[MACE-POLAR](https://github.com/ACEsuit/mace-foundations/releases/tag/mace_polar_1) includes
long-range electrostatics within the ML region. The ML/MM coupling here remains mechanical:
MM charges do not produce a field in the ML model, and its predicted charges do not update
the classical cross-region interactions. This interface does not expose an `external_field`
option. Whole-molecule and force-field restrictions above still apply.

## Jobs and dynamics

Use `MLMMTheory` with {func}`~openmmqmmm.single_point`,
{func}`~openmmqmmm.optimize_geometry`, {func}`~openmmqmmm.numerical_frequencies` and
{func}`~openmmqmmm.openmm_md`. Energies are in Hartree, gradients in Hartree/Bohr and input
coordinates in Angstrom, as for the other theories. No QM charge or multiplicity is
required; any electronic-state settings belong to the chosen model's supported options.

Single points, optimization and frequencies require the source MM theory to have
`autoconstraints=None` and `rigidwater=False`. By default `remove_constraints=True` removes
constraints within the ML region only; constraints in the MM environment remain. Removing
ML constraints usually requires a smaller MD timestep; the example uses 0.5 fs.

The mixed system runs through native OpenMM forces, including native RPMD when the ML
backend and OpenMM platform support it. For {doc}`rpmd`, start with no constraints anywhere
and physical masses (`hydrogenmass=None`). Backend support must be checked for the
particular model and platform. `rpmd_qm_num_copies` is a QM/MM option and does not contract
the ML forces. {class}`~openmmqmmm.OpenMMQMMMCalculator` and
{func}`~openmmqmmm.export_rpmd_potential` remain interfaces for `QMMMTheory`.

## Scope of the model

Mechanical embedding keeps classical cross-region interactions, including the force
field's fixed charges and van der Waals parameters. It does not provide polarization by the
MM environment, charge transfer between regions or reactions across the ML/MM boundary. A
model trained on isolated molecules does not acquire those capabilities merely by being
placed in solvent.

Check the model's supported elements, charge states, spin states, molecular sizes and
training domain. Validate energies, forces and the sampled configurations for the intended
system; choosing a smaller timestep cannot correct an unsuitable potential. The entire
selected subset is passed to one potential, so selecting several molecules also asks that
model to describe their mutual interactions.

A runnable script with explicit atom or residue selection is described in
{doc}`../examples`.

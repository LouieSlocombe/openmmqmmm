# Examples

Runnable scripts live in
[examples/](https://github.com/LouieSlocombe/openmmqmmm/tree/main/examples) in the
repository. Each one runs as it stands, given the inputs it names.

## gasphase_hf.py

```bash
python examples/gasphase_hf.py
```

A gas-phase ORCA calculation on HF: single point, geometry optimization and numerical
frequencies, one after the other on the same fragment. Needs ORCA and nothing else — no PDB
file, no force field — which makes it the right first thing to run against a new
installation. Walked through in {doc}`quickstart`.

## qmmm_optimization.py

```bash
python examples/qmmm_optimization.py system.pdb
```

Electrostatically embedded QM/MM: a CHARMM36 protein, an ORCA QM region of eight atoms, and a
geometry optimization restricted to that region. The shape of most QM/MM work — see
{doc}`guide/qmmm`.

## mace_scf_qmmm.py

```bash
python examples/mace_scf_qmmm.py /path/to/fixedpoint.model
python examples/mace_scf_qmmm.py /path/to/qeq.model --model-type qeq --device cuda
```

Electrostatically embedded MACE-SCF/MM on a generated, nonperiodic water dimer: the
first water is described by the model and the second by OpenMM TIP3P. The script computes
the energy and all atomic gradients, including forces from the MM charge field, and
writes `results_singlepoint.json`. Supply a checkpoint supporting neutral water and
the optional MACE-SCF dependencies from {doc}`install`; no PDB, ORCA or OpenMM-ML is
needed. `--platform` chooses the OpenMM platform independently of the model's `--device`.
This small example demonstrates the interface, not a validated water potential.

## mlmm_dynamics.py

```bash
python examples/mlmm_dynamics.py system.pdb \
    --forcefield amber14-all.xml amber14/tip3pfb.xml ligand.xml \
    --ml-residues 1 --potential ani2x --periodic --steps 1000
```

Mechanically embedded ML/MM: an initial single point followed by a short trajectory.
Supply a prepared PDB and force-field XML files covering the full system. `--ml-residues`
selects zero-based topology residue indices, not PDB residue IDs; alternatively use
`--ml-atoms` with zero-based atom indices. Select complete molecules supported by the
chosen model. The example needs OpenMM-ML, its model backend and weights, but no ORCA.
See `--help` for platform and device options and {doc}`guide/mlmm` for limitations.

For MACE-POLAR, supply the selected ML region's charge and multiplicity:

```bash
python examples/mlmm_dynamics.py system.pdb \
    --forcefield amber14-all.xml amber14/tip3pfb.xml ligand.xml \
    --ml-residues 1 --potential mace-polar --ml-charge -1 --ml-multiplicity 1 \
    --platform CUDA --device cuda --periodic --steps 1000
```

`mace-polar` selects the medium model. Use `--model-path /path/to/MACE-POLAR.model` for
a local polar checkpoint; `--precision` selects `single` or `double` (the polar default).
MM charges still couple mechanically to the ML region. See {doc}`install` for the
additional MACE-POLAR dependencies.

## qmmm_rpmd_nqe_stages.py

```bash
python examples/qmmm_rpmd_nqe_stages.py system.pdb
```

QM/MM ring-polymer dynamics driven through openmmnqe's staged RPMD workflow. Needs ORCA and
an environment holding both packages —  `build_tools/README.md` has the recipe. The same
pattern with an analytic QM stand-in instead of ORCA is exercised by
[tests/test_nqe_interop.py](https://github.com/LouieSlocombe/openmmqmmm/blob/main/tests/test_nqe_interop.py),
which runs without ORCA. See {doc}`guide/rpmd` and {doc}`guide/nqe_interop`.

## periodic_embedding_validation.py

```bash
python examples/periodic_embedding_validation.py --output periodic-validation.json
```

An analytic QM stand-in exercises the finite periodic embedding through OpenMM's
Reference platform without ORCA. It records cell-size and embedding-extent sensitivity
of relative energies and forces, plus short NVE trajectories with and without an image
switch. See {doc}`guide/periodic-validation` for the baseline interpretation and the
protocol for repeating these checks on a real QM/MM system.

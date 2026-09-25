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

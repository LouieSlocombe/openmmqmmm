# Embedding audit reproducers

These scripts preserve the numerical probes used in the embedding audit. They print observed results, including deliberately reproduced defects; successful execution does not mean every physical invariant passed. The recorded run is in [results.txt](results.txt).

The scripts and recorded results are historical snapshots of the audited revision `b4495f8`. Subsequent fixes validate unsupported configurations earlier, so some snapshot probes now stop with an intentional rejection before reaching all their calculations. The numerical probes and recorded output remain unchanged; the current expected behavior is covered by the `tests/test_embedding_*` regression suites.

Run a script using the project's development Python environment, for example:

```sh
python docs/audits/embedding-repros/boundary.py
```

An absolute script path also works from any working directory. Each script locates the checkout from its own file path and creates a fresh temporary working directory for generated OpenMM, Fragment, and ORCA files. That directory is removed on exit.

| Script | Probe |
| --- | --- |
| [boundary.py](boundary.py) | Adjacent MM1 redistribution, charge/dipole preservation, nonpolar-cut guard, and nonperiodic topology disagreement. |
| [mm.py](mm.py) | Exception parameter offsets, stale exception charge products, custom torsion stripping, and geometry-dependent population-charge forces. |
| [virtual_sites.py](virtual_sites.py) | Direct-call virtual-site gradients/coordinates and OpenMM force callback redistribution. |
| [periodic_residuals.py](periodic_residuals.py) | All-QM residual energies from dispersion correction, LJPME, and periodic Coulomb terms. |
| [orca_derivatives.py](orca_derivatives.py) | Real ORCA electrostatic embedding gradients against finite differences. |
| [orca_charge_updates.py](orca_charge_updates.py) | Real ORCA mechanical population-charge update interface; catches the expected InputError after QM finishes. |

The OpenMM probes use the Reference platform. The virtual-site probe also requires pytest because it imports the Coulomb test fixture using an absolute path derived from the checkout. The two ORCA probes require a working ORCA installation found through PATH or OPENMMQMMM_ORCADIR.

The newer [periodic imaging benchmark](periodic_imaging_benchmark.py) compares the
current implementation with a fixed Git baseline and measures complete fresh
Python QM/MM callbacks using an analytic backend. Its [report](periodic-imaging-performance.md)
and [raw samples](periodic-imaging-benchmark.json) are separate from the historical
audit snapshots above; those original reproducers and results remain unchanged.

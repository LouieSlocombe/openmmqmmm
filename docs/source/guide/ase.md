# ASE calculator

{class}`~openmmqmmm.OpenMMQMMMCalculator` exposes a configured
{class}`~openmmqmmm.QMMMTheory` to the
[Atomic Simulation Environment](https://ase-lib.org), so ASE's optimizers, dynamics and
analysis tools can drive a QM/MM potential.

```python
from ase import Atoms
from ase.optimize import BFGS
from openmmqmmm import OpenMMQMMMCalculator

atoms = Atoms(fragment.elems, positions=fragment.coords)
atoms.calc = OpenMMQMMMCalculator(qm_mm, directory="ase-qmmm")
BFGS(atoms).run(fmax=0.05)
```

Energies come back in eV and forces in eV/Å, as ASE expects — openmmqmmm's own job functions
use Hartree and Hartree/Bohr.

## What it requires

- The ASE `Atoms` object must keep the atom count, elements and ordering of the
  {class}`~openmmqmmm.Fragment` the QM/MM theory was built from. The calculator maps atom *i*
  to atom *i*; it does not match by element or position.
- Cell changes and stress are not supported. The QM/MM theory owns the periodicity.
- Charge and multiplicity are taken from `QMMMTheory.qm_charge` and `qm_mult`. If the theory
  does not carry them, pass `charge=` and `mult=` to the calculator instead.

Use a separate theory instance, process and calculation directory for every concurrent ASE
calculation. A theory object owns its scratch directory, and two calculations sharing one will
overwrite each other's files.

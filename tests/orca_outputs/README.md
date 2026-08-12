# ORCA reference outputs

Real ORCA 6.1.1 output for a water molecule, read by `test_orca_parsers.py`. The
parsers are pure text -> number functions, so committed output is what lets them be
tested without an ORCA installation — CI has none.

The `.inp`/`.pc` inputs are committed alongside the outputs on purpose. They were not,
originally, and neither was the output: `tests/.gitignore` starts with a `*` catch-all
and its whitelist never named `orca_outputs/`, so the whole directory was ignored in
silence, never reached a commit, and vanished when the untracked local copy was cleaned
away. Thirteen tests broke with no way to reconstruct the data. Keep the inputs here.

## Geometry

All three runs use the same water, in Angstrom — the geometry `WATER_COORDS` in
`test_freq.py` also uses:

```
O   0.0000000   0.0000000   0.1173000
H   0.0000000   0.7572000  -0.4692000
H   0.0000000  -0.7572000  -0.4692000
```

It is a standard experimental water geometry (r(OH) = 0.9572 A, angle 104.52 deg), not
an HF/def2-SVP stationary point, so the gradient is non-zero — which is what makes it
useful for testing gradient parsers.

## Files

| Run | Input | Outputs | Exercises |
| --- | --- | --- | --- |
| HF/def2-SVP gradient | `h2o_engrad.inp` | `h2o_engrad.out`, `h2o_engrad.engrad` | energy, gradient, Mulliken/Loewdin charges, dipole, SCF-convergence and timing parsers |
| BP86/def2-SVP gradient in a point-charge field | `h2o_pc.inp`, `h2o_pc.pc` | `h2o_pc.out`, `h2o_pc.pcgrad` | point-charge gradient parser, and the `pc_gradient` / `rij_coulomb_gradient` / `xc_gradient` timings the QM/MM path reports |
| HF/def2-SVP frequencies | `h2o_freq.inp` | `h2o_freq.out`, `h2o_freq.hess` | Hessian and IR-intensity parsers |

The point-charge run needs `def2/J` for RI-J, and DFT rather than HF, or ORCA never
prints the `RI-J Coulomb gradient` and `XC gradient` timings the tests look for.

The two point charges are a neutral +/-0.417 pair in the yz-plane at hydrogen-bonding
distance, standing in for a neighbouring MM water. Being in the yz-plane keeps the
x-component of the point-charge gradient exactly zero, which the test asserts.

## Regenerating

From this directory, with ORCA 6.1.1 on PATH:

```sh
orca h2o_engrad.inp > h2o_engrad.out
orca h2o_pc.inp     > h2o_pc.out
orca h2o_freq.inp   > h2o_freq.out
```

Then trim each `.out` down to its `INPUT FILE` banner — everything above it is the ORCA
logo and credits, and it carries the absolute working directory, which should not be
committed:

```sh
for f in h2o_engrad h2o_pc h2o_freq; do
    n=$(grep -n '^ *INPUT FILE *$' "$f.out" | head -1 | cut -d: -f1)
    tail -n +$((n - 1)) "$f.out" > "$f.trimmed" && mv "$f.trimmed" "$f.out"
done
```

The reference values asserted in `test_orca_parsers.py` are whatever ORCA printed in
these files. A different ORCA version, geometry or point-charge placement will change
them, so update the constants in that module together with the files.

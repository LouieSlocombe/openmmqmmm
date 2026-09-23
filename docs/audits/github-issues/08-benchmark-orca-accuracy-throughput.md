**Title:** Add reproducible accuracy and throughput benchmarks for ORCA QM/MM settings

Suggested labels: `benchmark`, `research`  
Suggested priority: Medium

The audit identified candidate settings but did not establish production speedups. Add a repeatable benchmark matrix over warm displaced snapshots, ORCA process counts, OpenMM resources, scratch storage and relevant numerical options.

RI-J/RIJCOSX are already enabled in many ORCA inputs; compare against NORI only where relevant. In the small PBE/def2-SVP trial, RI-J changed force relative L2 norms by 0.220–0.233% and the two-geometry energy difference by 0.000677 kcal/mol.

FMM worked through the current external-charge interface on a 26,131-atom protein fixture. Energy changed by 0.000530 kcal/mol and the maximum force-component change was 2.52e-6 Eh/bohr. Full standalone wall time increased from 10.120 to 11.455 s in the warm trial; a total speedup has not been shown.

Acceptance criteria:

- [ ] Include moving equilibrium and reactive geometries with fixed reference settings.
- [ ] Report repeated median/p95 wall times, SCF iterations and resource configuration.
- [ ] Compare relative energies and both QM and MM force RMS/max errors.
- [ ] Separate electronic runtime from total callback/MD time.
- [ ] Preserve intended electronic states and initially retain TightSCF.
- [ ] Publish cases where a candidate is slower as well as faster; do not change defaults without evidence.

References: [ORCA RI](https://www.faccts.de/docs/orca/6.1/manual/contents/essentialelements/RI.html), [ORCA FMM](https://www.faccts.de/docs/orca/6.1/manual/contents/multiscalesimulations/fmm.html). Dependency: the proposed “Expose structured QM/MM MD timing, SCF and cache statistics” work.


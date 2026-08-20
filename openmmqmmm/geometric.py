from __future__ import annotations

import contextlib
import logging
import logging.config
import os
import shutil
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TypeAlias

import numpy as np

import openmmqmmm.constants
from openmmqmmm.coords import (
    Fragment,
    _print_internal_coordinate_table,
    _qm_region_owner,
    check_charge_mult,
    fullindex_to_actindex,
    print_coords_for_atoms,
    write_coords_all,
    write_xyz_for_atoms,
    write_xyzfile,
)
from openmmqmmm.coords_pbc import (
    align_to_standard_orientation,
    cell_vectors_to_params,
    cell_volume,
    write_cif_file,
    write_poscar_file,
    write_xsf_file,
)
from openmmqmmm.exceptions import (
    FileFormatError,
    InputError,
    InternalError,
    MissingDependencyError,
    OpenMMQMMMError,
)
from openmmqmmm.freq import approximate_full_hessian_from_smaller, read_hessian, write_hessian
from openmmqmmm.numgrad import NumGrad
from openmmqmmm.openmm import OpenMMTheory
from openmmqmmm.qmmm import QMMMTheory
from openmmqmmm.results import Results
from openmmqmmm.utils import log_time_since, main_header, pygrep2, sub_header

logger = logging.getLogger(__name__)

ConstraintDict: TypeAlias = dict[str, list[Any]]
ConvergenceCriteria: TypeAlias = dict[str, float]
HessianOption: TypeAlias = np.ndarray | str | None
StrPath: TypeAlias = str | os.PathLike[str]
CalculationResult: TypeAlias = dict[str, float | np.ndarray]

_GEOMETRIC_LOGGING_LOCK = threading.RLock()


def _run_optimizer_without_reconfiguring_logging(run_optimizer: Any, arguments: Mapping[str, Any]) -> Any:
    """Run geomeTRIC without allowing its legacy INI to replace application logging."""
    # geomeTRIC hard-codes a process-global fileConfig call and this integration
    # also uses fixed output names. Serializing runs prevents concurrent callers
    # from capturing each other's temporary fileConfig wrapper or log handler.
    with _GEOMETRIC_LOGGING_LOCK:
        return _run_optimizer_with_isolated_logging(run_optimizer, arguments)


def _run_optimizer_with_isolated_logging(run_optimizer: Any, arguments: Mapping[str, Any]) -> Any:
    """Intercept one geomeTRIC fileConfig call while the integration lock is held."""
    original_file_config = logging.config.fileConfig
    expected_config = arguments.get("logIni")
    geometric_logger = logging.getLogger("geometric")
    original_level = geometric_logger.level
    original_disabled = geometric_logger.disabled
    original_effective_level = geometric_logger.getEffectiveLevel()
    run_handlers: list[logging.Handler] = []
    filtered_handlers: list[logging.Handler] = []

    def preserve_configured_level(record: logging.LogRecord) -> bool:
        if record.name == "geometric" or record.name.startswith("geometric."):
            return not original_disabled and record.levelno >= original_effective_level
        return True

    # The internal step log requires geomeTRIC INFO records even when the
    # application selected WARNING. Keep those newly-enabled records out of
    # pre-existing application handlers while allowing the run-only file
    # handler below to receive them.
    if original_disabled or original_effective_level > logging.INFO:
        application_handlers = set(geometric_logger.handlers)
        if geometric_logger.propagate:
            application_handlers.update(logging.getLogger().handlers)
        for handler in application_handlers:
            handler.addFilter(preserve_configured_level)
            filtered_handlers.append(handler)

    def preserve_application_logging(
        filename: Any,
        defaults: Mapping[str, Any] | None = None,
        disable_existing_loggers: bool = True,
        encoding: str | None = None,
    ) -> None:
        if filename == expected_config:
            # run_optimizer invokes fileConfig before doing any work. Restore the
            # function immediately so the process-wide patch lasts only until
            # that one known call, rather than for the optimization itself.
            logging.config.fileConfig = original_file_config
            logger.debug("Leaving geomeTRIC output under the application's logging configuration")
            log_filename = (defaults or {}).get("logfilename")
            if log_filename is not None:
                file_handler = logging.FileHandler(log_filename, encoding="utf-8")
                # geomeTRIC includes its own newlines and its original RawFileHandler
                # therefore adds no terminator of its own.
                file_handler.terminator = ""
                file_handler.setLevel(logging.INFO)
                file_handler.setFormatter(logging.Formatter("%(message)s"))
                geometric_logger.addHandler(file_handler)
                run_handlers.append(file_handler)
                geometric_logger.setLevel(min(original_effective_level, logging.INFO))
                geometric_logger.disabled = False
            return
        original_file_config(
            filename,
            defaults=defaults,
            disable_existing_loggers=disable_existing_loggers,
            encoding=encoding,
        )

    logging.config.fileConfig = preserve_application_logging
    try:
        return run_optimizer(**arguments)
    finally:
        if logging.config.fileConfig is preserve_application_logging:
            logging.config.fileConfig = original_file_config
        for handler in run_handlers:
            geometric_logger.removeHandler(handler)
            handler.close()
        for handler in filtered_handlers:
            handler.removeFilter(preserve_configured_level)
        geometric_logger.setLevel(original_level)
        geometric_logger.disabled = original_disabled


# Convergence thresholds by preset name. Every preset sets the same six, and cmax (the
# constraint violation) is 1.0e-2 throughout because it is a tolerance on the constraints
# themselves rather than on the optimisation.
#
# ORCA is the default. Chemshell and GAU carry identical numbers today; they are separate
# entries because they name different programs' defaults, and one changing upstream should
# not silently change the other.
_CONVERGENCE_KEYS = ("energy", "grms", "gmax", "drms", "dmax", "cmax")

# fmt: off
_CONVERGENCE_THRESHOLDS = {
    #                   energy, grms,   gmax,   drms,   dmax,   cmax
    "ORCA":            (5e-6,   1e-4,   3.0e-4, 2.0e-3, 4.0e-3, 1.0e-2),
    "ORCA_TIGHT":      (1e-6,   3e-5,   1.0e-4, 6.0e-4, 1.0e-3, 1.0e-2),
    "Chemshell":       (1e-6,   3e-4,   4.5e-4, 1.2e-3, 1.8e-3, 1.0e-2),
    "GAU":             (1e-6,   3e-4,   4.5e-4, 1.2e-3, 1.8e-3, 1.0e-2),
    "GAU_TIGHT":       (1e-6,   1e-5,   1.5e-5, 4.0e-5, 6e-5,   1.0e-2),
    "GAU_VERYTIGHT":   (1e-6,   1e-6,   2e-6,   4.0e-6, 6e-6,   1.0e-2),
    "SuperLoose":      (1e-1,   1e-1,   1e-1,   1e-1,   1e-1,   1.0e-2),
}
# fmt: on

#: Convergence thresholds passed to geomeTRIC, by preset name, in geomeTRIC's own keys.
CONVERGENCE_PRESETS = {
    name: {f"convergence_{key}": value for key, value in zip(_CONVERGENCE_KEYS, thresholds, strict=True)}
    for name, thresholds in _CONVERGENCE_THRESHOLDS.items()
}


@dataclass
class Constraints:
    """The geomeTRIC constraint lists: bonds, angles, dihedrals and Cartesian freezes."""

    bond: list | None = None
    angle: list | None = None
    dihedral: list | None = None
    xyz: list | None = None
    x: list | None = None
    y: list | None = None
    z: list | None = None
    xy: list | None = None
    xz: list | None = None
    yz: list | None = None


def optimize_geometry(
    *,
    theory: Any = None,
    fragment: Fragment | None = None,
    charge: int | None = None,
    mult: int | None = None,
    coordsystem: str = "tric",
    force_coordsystem: bool = False,
    frozenatoms: list[int] | None = None,
    constraints: ConstraintDict | None = None,
    constraintsinputfile: StrPath | None = None,
    irc: bool = False,
    rigid: bool = False,
    enforce_constraints: float | None = None,
    constrainvalue: bool = False,
    maxiter: int = 250,
    active_region: bool = False,
    actatoms: list[int] | None = None,
    num_grad: bool = False,
    convergence_setting: str | None = None,
    conv_criteria: Mapping[str, float] | None = None,
    print_atoms_list: list[int] | None = None,
    ts_opt: bool = False,
    hessian: HessianOption = None,
    partial_hessian_atoms: list[int] | None = None,
    modelhessian: str | None = None,
    subfrctor: float = 1,
    mm_pdb_traj_write: bool = False,
    result_write_to_disk: bool = True,
    force_no_pbc: bool = False,
    pbc_format_option: str = "CIF",
) -> Results:
    """Optimize the geometry of a fragment with a given theory, using geomeTRIC."""
    timeA = time.time()

    if theory is None or fragment is None:
        raise InputError("geomeTRICOptimizer requires theory and fragment objects provided. Exiting.")
    # NOTE: Class does not take fragment and theory
    optimizer = GeometricOptimizer(
        theory=theory,
        charge=charge,
        mult=mult,
        coordsystem=coordsystem,
        frozenatoms=frozenatoms,
        maxiter=maxiter,
        active_region=active_region,
        actatoms=actatoms,
        ts_opt=ts_opt,
        hessian=hessian,
        partial_hessian_atoms=partial_hessian_atoms,
        modelhessian=modelhessian,
        constraintsinputfile=constraintsinputfile,
        irc=irc,
        rigid=rigid,
        enforce_constraints=enforce_constraints,
        convergence_setting=convergence_setting,
        conv_criteria=conv_criteria,
        print_atoms_list=print_atoms_list,
        subfrctor=subfrctor,
        mm_pdb_traj_write=mm_pdb_traj_write,
        force_coordsystem=force_coordsystem,
        result_write_to_disk=result_write_to_disk,
        force_no_pbc=force_no_pbc,
        pbc_format_option=pbc_format_option,
    )

    if num_grad:
        logger.info("NumGrad flag detected. Wrapping theory object into NumGrad class")
        logger.info("This enables numerical-gradient calculation for theory")
        theory = NumGrad(theory=theory)

    result = optimizer.run(
        theory=theory,
        fragment=fragment,
        charge=charge,
        mult=mult,
        constraints=constraints,
        constrainvalue=constrainvalue,
    )
    log_time_since(timeA, "geomeTRIC")

    return result


class GeometricOptimizer:
    """Geometry optimizer wrapping the geomeTRIC library."""

    def __init__(
        self,
        *,
        theory: Any = None,
        charge: int | None = None,
        mult: int | None = None,
        coordsystem: str = "tric",
        frozenatoms: list[int] | None = None,
        maxiter: int = 250,
        active_region: bool = False,
        actatoms: list[int] | None = None,
        convergence_setting: str | None = None,
        conv_criteria: Mapping[str, float] | None = None,
        ts_opt: bool = False,
        hessian: HessianOption = None,
        constraintsinputfile: StrPath | None = None,
        irc: bool = False,
        rigid: bool = False,
        enforce_constraints: float | None = None,
        print_atoms_list: list[int] | None = None,
        partial_hessian_atoms: list[int] | None = None,
        modelhessian: str | None = None,
        subfrctor: float = 1,
        mm_pdb_traj_write: bool = False,
        force_coordsystem: bool = False,
        result_write_to_disk: bool = True,
        force_no_pbc: bool = False,
        pbc_format_option: str = "CIF",
    ) -> None:
        import time

        self.time_init = time.time()
        logger.info(main_header("geomeTRICOptimizer initialization"))
        logger.debug("Creating optimizer object")

        if actatoms is not None:
            logger.info("List of active atoms provided. Setting ActiveRegion to True")
            active_region = True
        if actatoms is None:
            actatoms = []
        if frozenatoms is None:
            frozenatoms = []

        if active_region is True and coordsystem.lower() == "tric":
            logger.warning(
                "ActiveRegion is set but the coordsystem is TRIC. The HDLC coordinate system is usually much "
                "more robust for large systems than TRIC."
            )
            if force_coordsystem is True:
                logger.info("force_coordsystem is True.")
                logger.info("Sticking with coordsystem TRIC")
            else:
                logger.info("force_coordsystem is False.")
                logger.warning(
                    "Switching to HDLC to avoid likely robustness problems with TRIC. To avoid this "
                    "behaviour (and force use of TRIC) you can use set the Boolean force_coordsystem to True."
                )
                coordsystem = "hdlc"

        self.maxiter = maxiter
        self.actatoms = actatoms
        self.frozenatoms = frozenatoms
        self.coordsystem = coordsystem
        self.print_atoms_list = print_atoms_list
        self.active_region = active_region
        self.ts_opt = ts_opt
        self.subfrctor = subfrctor

        self.irc = irc
        self.rigid = rigid
        self.enforce_constraints = enforce_constraints

        self.mm_pdb_traj_write = mm_pdb_traj_write
        self.hessian = hessian
        self.modelhessian = modelhessian
        self.partial_hessian_atoms = partial_hessian_atoms

        self.constraints = None
        # Optional user-constraintsfile in geometric syntax
        self.constraintsinputfile = constraintsinputfile

        self.result_write_to_disk = result_write_to_disk

        # Setup convergence criteria (sets self.conv_criteria)
        self.convergence_criteria(convergence_setting, conv_criteria)

        if getattr(theory, "periodic", False):
            logger.info("Detected periodicity in Theory object")
            logger.info("Activating periodic routines ")
            self.pbc_active = True
            self.pbc_format_option = pbc_format_option
            logger.info("Switching coordsystem to hdlc")
            self.coordsystem = "hdlc"
            logger.info("Final PBC coordinate file written in format: %s", self.pbc_format_option)

            if force_no_pbc is True:
                logger.warning("Option force_noPBC set to True. Turning off PBC")
                self.pbc_active = False
        else:
            logger.info("Theory is not periodic")
            self.pbc_active = False

        logger.info("Coordinate system:  %s", self.coordsystem)
        logger.info("Max iterations:  %s", self.maxiter)
        logger.info("Frozen atoms: %s", self.frozenatoms)
        logger.info("Active Region: %s", self.active_region)
        if self.active_region is True:
            logger.info("Number of active atoms: %s", len(self.actatoms))
        logger.info("TS Optimization: %s", self.ts_opt)
        logger.info("Hessian Option: %s", self.hessian)
        logger.info("Convergence criteria: %s", self.conv_criteria)

    def print_atoms_output_setting(self, theory: Any, fragment: Fragment) -> None:
        # What atoms to print in outputfile in each opt-step. Example choice: QM-region only
        # If not specified then active-region or all-atoms
        """Decide which atoms are printed in each optimization step's output."""
        if self.print_atoms_list is None:
            if self.active_region is True:
                if isinstance(theory, QMMMTheory):
                    logger.info("Theory class: QMMMTheory")
                    logger.debug(
                        "Will by default print only QM-region in output (use print_atoms_list option to change)"
                    )
                    self.print_atoms_list = theory.qmatoms
                else:
                    # Print actatoms since using Active Region (can be too much)
                    self.print_atoms_list = self.actatoms
            else:
                self.print_atoms_list = fragment.allatoms

    def convergence_criteria(self, convergence_setting: str | None, userconv: Mapping[str, float] | None) -> None:
        """Resolve the geomeTRIC convergence thresholds to use."""
        if convergence_setting is None:
            if userconv is None:
                logger.debug("No convergence settings by user. Using default criteria (same as ORCA)")
            convergence_setting = "ORCA"
        if convergence_setting not in CONVERGENCE_PRESETS:
            raise InputError(
                f"Unknown convergence setting '{convergence_setting}'. "
                f"Expected one of: {', '.join(CONVERGENCE_PRESETS)}."
            )

        self.conv_criteria = dict(CONVERGENCE_PRESETS[convergence_setting])
        if userconv is not None:
            logger.info("User-defined convergence criteria:")
            self.conv_criteria.update(userconv)

    def define_constraints(self, constraints: ConstraintDict | None) -> Constraints:
        """Translate the user constraints dict into geomeTRIC's constraint lists."""
        logger.debug("Defining constraints: %s", constraints)
        # For QM/MM we need to convert full-system atoms into active region atoms
        if self.active_region and constraints is not None:
            logger.info("Constraints set. Active region true")
            logger.info("User-defined constraints (fullsystem-indices): %s", constraints)
            constraints = _constraints_indices_convert(constraints, self.actatoms)
            logger.info("Converting constraints indices to active-region indices")
            logger.info("Constraints (actregion-indices): %s", constraints)

        if constraints is None:
            return Constraints()
        return Constraints(
            bond=constraints.get("bond"),
            angle=constraints.get("angle"),
            # geomeTRIC calls these dihedrals; both spellings are accepted from the user
            dihedral=constraints.get("dihedral", constraints.get("torsion")),
            xyz=constraints.get("xyz"),
            x=constraints.get("x"),
            y=constraints.get("y"),
            z=constraints.get("z"),
            xy=constraints.get("xy"),
            xz=constraints.get("xz"),
            yz=constraints.get("yz"),
        )

    def write_constraintsfile(self, frozenatoms: Sequence[int], constraints: Constraints, constrainvalue: bool) -> None:
        """Write the geomeTRIC constraints.txt file."""
        logger.debug("Writing constraints file")

        with contextlib.suppress(FileNotFoundError):
            os.remove("constraints.txt")

        # geomeTRIC keyword and number of atom indices, per constraint kind. The Cartesian
        # freezes take one index and never a value; the internal coordinates take a target
        # value as a final element when constrainvalue is set.
        sections = [
            ("bond", "distance", 2),
            ("angle", "angle", 3),
            ("dihedral", "dihedral", 4),
            ("x", "x", 1),
            ("y", "y", 1),
            ("z", "z", 1),
            ("xy", "xy", 1),
            ("xz", "xz", 1),
            ("yz", "yz", 1),
        ]

        lines = []
        if len(frozenatoms) > 0:
            logger.info("Writing frozen atom constraints")
            lines.append("$freeze")
            # Changing from zero-indexing (openmmqmmm) to 1-indexing (geomeTRIC)
            lines += [f"xyz {atom + 1}" for atom in frozenatoms]

        for attribute, keyword, num_indices in sections:
            entries = getattr(constraints, attribute)
            if entries is None:
                continue
            logger.info("Writing %s constraints %s", attribute, entries)
            # A Cartesian freeze has no value to set, so it stays under $freeze either way
            with_value = constrainvalue is True and num_indices > 1
            lines.append("$set" if with_value else "$freeze")
            for entry in entries:
                # Changing from zero-indexing (openmmqmmm) to 1-indexing (geomeTRIC)
                indices = [entry + 1] if num_indices == 1 else [i + 1 for i in entry[:num_indices]]
                fields = [keyword, *(str(i) for i in indices)]
                if with_value:
                    fields.append(str(entry[num_indices]))
                lines.append(" ".join(fields))

        self.constraintsfile = None
        if lines:
            self.constraintsfile = "constraints.txt"
            with open("constraints.txt", "w") as confile:
                confile.write("\n".join(lines) + "\n")

    def cleanup(self) -> None:
        """Delete the optimizer's scratch files, including any constraints.txt, before a run."""
        tmpfiles = [
            "geometric_OPTtraj.log",
            "geometric_OPTtraj.xyz",
            "geometric_OPTtraj_Full.xyz",
            "geometric_OPTtraj_QMregion.xyz",
            "optimization_energies.log",
            "constraints.txt",
            "initialxyzfiletric.xyz",
            "geometric_OPTtraj.tmp",
            "dummyprefix.tmp",
            "dummyprefix.log",
            "fragment_optimized.frag",
            "Fragment-optimized.xyz",
            "Fragment-optimized_Active.xyz",
            "geometric_OPTtraj-PDB.pdb",
        ]
        for tmpfile in tmpfiles:
            # rmtree for the directories in the list, os.remove for the plain files it
            # raises NotADirectoryError on. Absent entries are fine: this is a pre-run tidy.
            try:
                shutil.rmtree(tmpfile)
            except FileNotFoundError:  # noqa: PERF203 - a file system call per entry dwarfs the try
                pass
            except NotADirectoryError:
                os.remove(tmpfile)

    def hessian_option(
        self,
        fragment: Fragment,
        actatoms: list[int],
        theory: Any,
        charge: int | None,
        mult: int | None,
        modelhessian: str | None,
    ) -> None:
        """Provide the starting Hessian geomeTRIC was asked for."""
        atomsused = fragment.allatoms if len(actatoms) == 0 else actatoms

        if isinstance(self.hessian, np.ndarray):
            logger.info("Hessian option provided is a Numpy array.")

            logger.info("Checking that Hessian is compatible with active atoms")
            if self.hessian.shape[0] != 3 * len(atomsused):
                raise InputError(
                    "{}\n{}".format(
                        f"Error: Hessian shape is {self.hessian.shape}  which is incompatible with the  number of "
                        f"active atoms present ({len(atomsused)})",
                        f"Hessian should have dimension of 3*N x 3*N where N is the number of active-atoms of the "
                        f"system (should be : {3 * len(atomsused)} x {3 * len(atomsused)})",
                    )
                )

            logger.info("Writing Hessian array to disk.")

            hessianfile = "Hessian_np"
            write_hessian(self.hessian, hessfile=hessianfile)
            self.hessian = "file:" + hessianfile
            logger.info("Hessian option to be used by geometric: %s", self.hessian)
        elif isinstance(self.hessian, str):
            logger.info("Hessian option provided is a string")
            if self.hessian == "xtb":
                raise InputError(
                    "Error: hessian='xtb' is not available in this ORCA+OpenMM build. Use '1point', '2point', "
                    "'partial' or a Hessian file instead."
                )
            if self.hessian == "1point":
                logger.info("Requested Hessian from Numfreq 1-point approximation (running in serial)")
                result_freq = openmmqmmm.numerical_frequencies(
                    theory=theory,
                    fragment=fragment,
                    charge=charge,
                    mult=mult,
                    npoint=1,
                    runmode="serial",
                    numcores=theory.numcores,
                )
                hessianfile = "Hessian_from_theory"
                shutil.copyfile("Numfreq_dir/Hessian", hessianfile)
                self.hessian = "file:" + str(hessianfile)
            elif self.hessian == "2point":
                logger.info("Requested Hessian from Numfreq 2-point approximation (running in serial)")
                result_freq = openmmqmmm.numerical_frequencies(
                    theory=theory,
                    fragment=fragment,
                    charge=charge,
                    mult=mult,
                    npoint=2,
                    runmode="serial",
                    numcores=theory.numcores,
                )
                hessianfile = "Hessian_from_theory"
                shutil.copyfile("Numfreq_dir/Hessian", hessianfile)
                self.hessian = "file:" + str(hessianfile)
            elif self.hessian == "partial":
                logger.info("Partial Hessian option requested")

                if self.partial_hessian_atoms is None:
                    raise InputError(
                        "hessian='partial' option requires setting the partial_hessian_atoms option. Exiting."
                    )

                logger.info("Now doing partial Hessian calculation using atoms: %s", self.partial_hessian_atoms)
                # Note: hardcoding runmode='serial' for now
                result_freq = openmmqmmm.numerical_frequencies(
                    theory=theory,
                    fragment=fragment,
                    charge=charge,
                    mult=mult,
                    npoint=1,
                    hessatoms=self.partial_hessian_atoms,
                    runmode="serial",
                    numcores=1,
                )
                # Large Hessian is the actatoms Hessian if actatoms provided

                combined_hessian = approximate_full_hessian_from_smaller(
                    fragment,
                    result_freq.hessian,
                    self.partial_hessian_atoms,
                    large_atomindices=actatoms,
                    rest_hessian=modelhessian,
                )

                hessianfile = "Hessian_from_partial"
                write_hessian(combined_hessian, hessfile=hessianfile)
                self.hessian = "file:" + hessianfile
            elif self.hessian == "partial2":
                logger.info("Partial Numpoint=2 Hessian option requested")

                if self.partial_hessian_atoms is None:
                    raise InputError(
                        "hessian='partial' option requires setting the partial_hessian_atoms option. Exiting."
                    )

                logger.info("Now doing partial Hessian calculation using atoms: %s", self.partial_hessian_atoms)
                # Note: hardcoding runmode='serial' for now
                result_freq = openmmqmmm.numerical_frequencies(
                    theory=theory,
                    fragment=fragment,
                    charge=charge,
                    mult=mult,
                    npoint=2,
                    hessatoms=self.partial_hessian_atoms,
                    runmode="serial",
                    numcores=1,
                )
                # Large Hessian is the actatoms Hessian if actatoms provided

                combined_hessian = approximate_full_hessian_from_smaller(
                    fragment,
                    result_freq.hessian,
                    self.partial_hessian_atoms,
                    large_atomindices=actatoms,
                    rest_hessian=modelhessian,
                )

                hessianfile = "Hessian_from_partial"
                write_hessian(combined_hessian, hessfile=hessianfile)
                self.hessian = "file:" + hessianfile
            elif "file:" in self.hessian:
                hessianfile = self.hessian.replace("file:", "")

            # Allow first and each options still
            if self.hessian not in ["first", "each"]:
                logger.info("Checking that defined Hessian is compatible with active-region")
                hessian_read = read_hessian(hessianfile)
                logger.info("actatoms: %s", actatoms)
                if hessian_read.shape[0] != 3 * len(atomsused):
                    raise InputError(
                        "{}\n{}".format(
                            f"Error: Hessian shape is {hessian_read.shape}  which is incompatible with the  number of "
                            f"active atoms present ({len(atomsused)})",
                            f"Hessian should have dimension of 3*N x 3*N where N is the number of active-atoms of the "
                            f"system (should be : {3 * len(atomsused)} x {3 * len(atomsused)})",
                        )
                    )
        elif self.hessian is None:
            logger.info("No Hessian option provided.")
        else:
            raise InputError("Unknown Hessian option")

    def setup_active_region_geometry(self, fragment: Fragment) -> None:
        """Build the reduced geometry and topology for an active-region optimization."""
        if len(self.actatoms) == 0:
            raise InputError("Error: List of active atoms (actatoms) provided is empty. This is not allowed.")
        # Sorting list, otherwise trouble
        self.actatoms.sort()
        logger.info("Active Region option Active. Passing only active-region coordinates to geomeTRIC.")
        logger.info("Active atoms list: %s", self.actatoms)
        logger.info("Number of active atoms: %s", len(self.actatoms))

        largest_atom_index = max(self.actatoms)
        if largest_atom_index >= fragment.numatoms:
            raise InputError(
                "{}\nThis does not make sense. Please provide a correct actatoms list. Exiting.".format(
                    f"Found active-atom index ({largest_atom_index}) that is larger or equal (>=) than the number of "
                    f"atoms of system ({fragment.numatoms})!"
                )
            )
        actcoords, actelems = fragment.get_coords_for_atoms(self.actatoms)

        # Writing act-region coords (only) of fragment to disk as XYZ file and reading into geomeTRIC
        write_xyzfile(actelems, actcoords, "initialxyzfiletric")

    def run(
        self,
        theory: Any = None,
        fragment: Fragment | None = None,
        charge: int | None = None,
        mult: int | None = None,
        constraints: ConstraintDict | None = None,
        constrainvalue: bool = False,
    ) -> Results:
        """Optimize a geometry with geomeTRIC."""
        logger.info(sub_header("Running geomeTRIC object"))
        logger.debug(
            f"\nDoing geometry optimization on fragment. Formula: {fragment.prettyformula} Label: {fragment.label} "
        )
        self.cleanup()  # NOTE: This deletes constraintsfile

        charge, mult = check_charge_mult(charge, mult, theory.theorytype, fragment, "geomeTRICOptimizer", theory=theory)
        # For QM/MM the resolved values describe the QM region, not this whole-system fragment, and
        # an MM theory resolves both to None. Stamping either onto the fragment would be written out
        # by write_xyzfile/print_system below and read back as the fragment's own charge.
        if charge is not None and mult is not None and _qm_region_owner(theory) is None:
            fragment.charge = charge
            fragment.mult = mult

        # If constraints not directly provided to run method, then we look at self.constraints and then
        # fragment.constraints
        if constraints is None:
            logger.debug("No constraints provided to run method.")
            logger.debug("Testing whether constraints are present in optimizer object")
            if self.constraints is not None:
                logger.debug("Found constraints in optimizer object")
                constraints = self.constraints
                constrainvalue = self.constrainvalue
            else:
                logger.debug("No constraints in optimizer object.")
                logger.debug("Now testing if constraints in fragment object ")
                if fragment.constraints is not None:
                    # Option used by Surface-scan relaxed parallel
                    logger.debug("Found constraints in fragment object")
                    constraints = fragment.constraints
                    constrainvalue = True  # Assuming to be the case.
                else:
                    logger.debug("No constraints in fragment object.")
        else:
            logger.info("Constraints provided to run method.")
        logger.info("\nConstraints:  %s", constraints)
        logger.info("constrainvalue:  %s", constrainvalue)
        parsed_constraints = self.define_constraints(constraints)
        if parsed_constraints.xyz is not None:
            logger.info("xyzconstraints found. Adding to frozenatoms")
            self.frozenatoms = self.frozenatoms + parsed_constraints.xyz
        self.write_constraintsfile(self.frozenatoms, parsed_constraints, constrainvalue)
        if self.constraintsinputfile is not None:
            logger.info("constraintsinputfile provided: %s", self.constraintsinputfile)
            if os.path.isfile(self.constraintsinputfile) is False:
                raise FileFormatError(f"Error:File {self.constraintsinputfile} does not exist")
            self.constraintsfile = self.constraintsinputfile

        if fragment.numatoms == 1:
            logger.info("System contains 1 atom, optimization makes no sense.")
            logger.info("Doing single-point energy calculation instead")
            return openmmqmmm.single_point(fragment=fragment, theory=theory, charge=charge, mult=mult)

        # ActiveRegion option where geomeTRIC only sees the QM part that is being optimized
        if self.active_region is True:
            self.setup_active_region_geometry(fragment)
        else:
            fragment.write_xyzfile("initialxyzfiletric.xyz")

        self.print_atoms_output_setting(theory, fragment)
        self.hessian_option(fragment, self.actatoms, theory, charge, mult, self.modelhessian)

        try:
            import geometric
        except Exception as e:
            raise MissingDependencyError(
                f"Problem importing geomeTRIC module!\nEither install geomeTRIC using pip:\n conda install geometric\n "
                f"or \n pip install geometric\n or manually from Github (https://github.com/leeping/geomeTRIC)\nActual "
                f"error message: {e}"
            ) from e
        # bondorders
        # generally unused, except PBC
        self.bothre = 0.0

        if self.pbc_active is True:
            logger.info("For PBC we activate constraints")
            self.bothre = 0.5
        mol_geometric_frag = geometric.molecule.Molecule("initialxyzfiletric.xyz")

        engine = GeometricEngine(
            mol_geometric_frag,
            theory,
            active_region=self.active_region,
            actatoms=self.actatoms,
            print_atoms_list=self.print_atoms_list,
            mm_pdb_traj_write=self.mm_pdb_traj_write,
            charge=charge,
            mult=mult,
            conv_criteria=self.conv_criteria,
            fragment=fragment,
            maxiter=self.maxiter,
            pbc_active=self.pbc_active,
        )
        logger.debug("Constraints file: %s", self.constraintsfile)
        final_geometric_args = GeometricArgs(
            engine,
            self.constraintsfile,
            coordsys=self.coordsystem,
            maxiter=self.maxiter,
            conv_criteria=self.conv_criteria,
            transition=self.ts_opt,
            hessian=self.hessian,
            subfrctor=self.subfrctor,
            verbose=0,
            irc=self.irc,
            rigid=self.rigid,
            enforce_constraints=self.enforce_constraints,
            bothre=self.bothre,
        )

        logger.info("Convergence criteria: %s", self.conv_criteria)
        logger.info("Hessian option: %s", self.hessian)
        logger.info("Coordinate system: %s", self.coordsystem)

        if self.ts_opt:
            logger.debug("Starting saddlepoint optimization")
        else:
            logger.debug("Starting optimization")

        log_time_since(self.time_init, "Time spent before run_optimizer")
        _run_optimizer_without_reconfiguring_logging(geometric.optimize.run_optimizer, vars(final_geometric_args))
        time.sleep(1)

        logger.info(f"\ngeomeTRIC Geometry optimization converged in {engine.iteration_count + 1} steps!\n")

        # QM/MM: Doing final energy evaluation if Truncated PC option was on
        if isinstance(theory, QMMMTheory):
            if theory.truncated_pc is True:
                logger.info(
                    "Truncated PC approximation was active. Doing final energy calculation with full PC environment"
                )
                theory.truncated_pc = False
                finalenergy, _finalgrad = theory.run(
                    current_coords=engine.full_current_coords,
                    elems=fragment.elems,
                    grad=True,
                    charge=charge,
                    mult=mult,
                )
            else:
                finalenergy = engine.energy
        else:
            finalenergy = engine.energy

        logger.info("Final optimized energy: %s", finalenergy)

        fragment.replace_coords(fragment.elems, engine.full_current_coords, conn=False)
        fragment.print_system(filename="fragment_optimized.frag")
        fragment.write_xyzfile(xyzfilename="Fragment-optimized.xyz")
        fragment.set_energy(finalenergy)

        if self.active_region is not True:
            logger.info("Final geometry")
            fragment.print_coords()

        if self.pbc_active:
            logger.info("PBC True. Writing final optimized geometry in PBC-format")
            logger.info("PBC_format_option: %s", self.pbc_format_option)
            if self.pbc_format_option.upper() == "CIF":
                convert_to_pbcfile = write_cif_file
                file_ext = "cif"
            elif self.pbc_format_option.upper() == "XSF":
                convert_to_pbcfile = write_xsf_file
                file_ext = "xsf"
            elif self.pbc_format_option.upper() == "POSCAR":
                convert_to_pbcfile = write_poscar_file
                file_ext = "POSCAR"
            convert_to_pbcfile(
                fragment.coords,
                fragment.elems,
                cellvectors=theory.periodic_cell_vectors,
                filename=f"Fragment-optimized.{file_ext}",
            )
            logger.info(f"Final cell vectors (Å):{theory.periodic_cell_vectors}")
            logger.info(f"Final cell parameters: ({cell_vectors_to_params(theory.periodic_cell_vectors)})")
            logger.info(f"Final cell volume (Å³):{cell_volume(theory.periodic_cell_vectors)}")
        if self.active_region is True:
            write_xyz_for_atoms(fragment.coords, fragment.elems, self.actatoms, "Fragment-optimized_Active")
        if isinstance(theory, QMMMTheory):
            write_xyz_for_atoms(fragment.coords, fragment.elems, theory.qmatoms, "Fragment-optimized_QMregion")

        if len(self.print_atoms_list) < 50:
            _print_internal_coordinate_table(fragment, actatoms=self.print_atoms_list)

        # Note: could include the geometry in object but can be very large causing printing head-aches on screen,
        # ignoring for now since the geometry is in the Fragment object anyway
        result = Results(label="Optimizer", energy=finalenergy)
        if self.result_write_to_disk is True:
            result.write_to_disk(filename="results_optimizer.json")
        return result


class GeometricArgs:
    """Argument bundle handed to geomeTRIC as ``**vars()``; the attribute names are geomeTRIC's."""

    def __init__(
        self,
        eng: GeometricEngine,
        constraintsfile: StrPath | None,
        *,
        coordsys: str,
        maxiter: int,
        conv_criteria: Mapping[str, float],
        transition: bool,
        hessian: HessianOption,
        subfrctor: float,
        verbose: int,
        irc: bool,
        rigid: bool,
        enforce_constraints: float | None,
        bothre: float,
    ) -> None:
        self.coordsys = coordsys
        self.maxiter = maxiter
        self.transition = transition
        self.hessian = hessian
        self.subfrctor = subfrctor
        self.verbose = verbose
        self.irc = irc
        self.rigid = rigid
        self.bothre = bothre
        if self.rigid is True:
            logger.info("Rigid optimization enabled.")
            logger.info("Activating revised constraint algorithm")
            self.conmethod = 1
        if enforce_constraints is not None:
            logger.info("enforce_constraints value passed: %s", enforce_constraints)
            self.enforce = enforce_constraints

        # Setting these to be part of kwargs that geometric reads
        self.convergence_energy = conv_criteria["convergence_energy"]
        self.convergence_grms = conv_criteria["convergence_grms"]
        self.convergence_gmax = conv_criteria["convergence_gmax"]
        self.convergence_drms = conv_criteria["convergence_drms"]
        self.convergence_dmax = conv_criteria["convergence_dmax"]
        self.convergence_cmax = conv_criteria["convergence_cmax"]
        self.prefix = "geometric_OPTtraj"
        self.input = "dummyinputname"
        self.constraints = constraintsfile
        # geomeTRIC requires a logging-config path in its argument object. The
        # runner above intercepts that config so it cannot replace the process's
        # root handlers, while retaining the path for upstream compatibility.
        path = openmmqmmm.constants.PACKAGE_DIR
        self.logIni = path + "/log.ini"
        self.customengine = eng


class GeometricEngine:
    """Adapter presenting an openmmqmmm theory as a geomeTRIC engine (energies and gradients)."""

    def __init__(
        self,
        geometric_molf: Any,
        theory: Any,
        *,
        active_region: bool = False,
        actatoms: list[int] | None = None,
        print_atoms_list: list[int] | None = None,
        charge: int | None = None,
        mult: int | None = None,
        conv_criteria: ConvergenceCriteria | None = None,
        fragment: Fragment | None = None,
        mm_pdb_traj_write: bool = False,
        maxiter: int | None = None,
        pbc_active: bool = False,
    ) -> None:
        # MM_PDB_traj_write on/off. Can be pretty big files
        self.mm_pdb_traj_write = mm_pdb_traj_write
        # Defining M attribute of engine object as geomeTRIC Molecule object
        self.M = geometric_molf
        self.theory = theory
        self.active_region = active_region
        # Defining current_coords for full system (not only act region)
        self.full_current_coords = []
        self.iteration_count = 0

        self.maxiter = maxiter
        self.energy = 0
        self.actatoms = actatoms
        self.print_atoms_list = print_atoms_list
        self.charge = charge
        self.mult = mult
        self.conv_criteria = conv_criteria
        self.fragment = fragment

        self.BOmatrix = None

        self.pbc_active = pbc_active
        if self.pbc_active is True:
            # Real elements
            self.elems_phys = self.fragment.elems
            aligned_atom_coords, aligned_vectors = align_to_standard_orientation(
                self.fragment.coords, theory.periodic_cell_vectors
            )
            self.fragment.coords = aligned_atom_coords
            self.theory.update_cell(aligned_vectors)

            self.H_ref = aligned_vectors.copy()
            self.H_ref_inv = np.linalg.inv(self.H_ref)

            # Modifying self.M to have aligned coords and 4 dummyatoms
            self.M.xyzs = [np.concatenate((aligned_atom_coords, [[0.0, 0.0, 0.0]], aligned_vectors), axis=0)]
            self.M.elem = [*self.M.elem, "F", "F", "F", "F"]

    def load_guess_files(self, dirname: StrPath) -> None:
        logger.debug("geomeTRIC called unsupported load_guess_files callback; continuing")

    def save_guess_files(self, dirname: StrPath) -> None:
        logger.debug("geomeTRIC called unsupported save_guess_files callback; continuing")

    # Optimizer may call this to see if the engine class is doing DFT with grid to print warning
    def detect_dft(self) -> bool:
        logger.debug("geomeTRIC called detect_dft callback")
        return True

    # geometric checks if calc_bondorder method is implemented for the custom engine. Disabled until we implement this
    def calc_bondorder(self, coords: np.ndarray, dirname: StrPath) -> np.ndarray | None:
        logger.debug("geomeTRIC called calc_bondorder callback")
        if self.BOmatrix is not None:
            return self.BOmatrix
        logger.debug("no BOmatrix found")
        if self.pbc_active:
            logger.info("PBC and BOmatrix handling")
            self.BOmatrix = np.zeros((len(self.M.elem), len(self.M.elem)), dtype=int)
            self.fragment.calc_connectivity()
            from openmmqmmm.coords import get_connected_atoms_dict

            conndict = get_connected_atoms_dict(self.fragment.coords, self.fragment.elems, 1.0, 0.1)
            logger.info("conndict: %s", conndict)
            for i, conn in conndict.items():
                for c in conn:
                    self.BOmatrix[i, c] = self.BOmatrix[c, i] = 1.0

            # Connecting origin and lattice atoms
            n_orig = len(self.elems_phys)
            self.BOmatrix[n_orig, n_orig + 1] = self.BOmatrix[n_orig + 1, n_orig] = 1
            self.BOmatrix[n_orig, n_orig + 2] = self.BOmatrix[n_orig + 2, n_orig] = 1
            self.BOmatrix[n_orig, n_orig + 3] = self.BOmatrix[n_orig + 3, n_orig] = 1

            return self.BOmatrix
        logger.debug("No BO option implemented")
        return None

    def clearCalcs(self) -> None:  # noqa: N802 - geomeTRIC engine API, do not rename
        logger.debug("geomeTRIC called unsupported clearCalcs callback; continuing")

    # Writing out trajectory file for full system in case of ActiveRegion. Note: Actregion coordinates are done done by
    # GeomeTRIC
    def write_trajectory_full(self) -> None:
        logger.info("Writing trajectory for Full system to file: geometric_OPTtraj_Full.xyz")
        with open("geometric_OPTtraj_Full.xyz", "a") as trajfile:
            trajfile.write(str(self.fragment.numatoms) + "\n")
            trajfile.write(f"Iteration {self.iteration_count} Energy {self.energy} \n")
            trajfile.writelines(
                el + "  " + str(cor[0]) + " " + str(cor[1]) + " " + str(cor[2]) + "\n"
                for el, cor in zip(self.fragment.elems, self.full_current_coords, strict=False)
            )

    def write_trajectory_qmregion(self) -> None:
        logger.info("Writing trajectory for QM-region to file: geometric_OPTtraj_QMregion.xyz")
        with open("geometric_OPTtraj_QMregion.xyz", "a") as trajfile:
            trajfile.write(str(len(self.theory.qmatoms)) + "\n")
            trajfile.write(f"Iteration {self.iteration_count} Energy {self.energy} \n")
            qm_coords, qm_elems = self.fragment.get_coords_for_atoms(self.theory.qmatoms)
            trajfile.writelines(
                el + "  " + str(cor[0]) + " " + str(cor[1]) + " " + str(cor[2]) + "\n"
                for el, cor in zip(qm_elems, qm_coords, strict=False)
            )

    def write_energy_logfile(self) -> None:
        logger.info("Writing logfile with energies: optimization_energies.log")
        with open("optimization_energies.log", "a") as trajfile:
            if self.iteration_count == 0:
                trajfile.write("Iteration QM-energy       (Eh) MM-Energy (Eh)  QM/MM-Energy (Eh)\n")
            trajfile.write(
                f"{self.iteration_count}         {self.theory.QMenergy} {self.theory.MMenergy} "
                f"{self.theory.QM_MM_energy}\n"
            )

    def write_pdbtrajectory(self) -> None:
        logger.info("Writing PDB-trajectory to file: geometric_OPTtraj-PDB.pdb")
        pdbtrajectoryfile = "geometric_OPTtraj-PDB.pdb"
        # STILL problem with PBC
        state = self.theory.mm_theory.simulation.context.getState(
            getEnergy=False, getPositions=True, getForces=False, enforcePeriodicBox=True
        )
        newpos = state.getPositions()
        with open(pdbtrajectoryfile, "a") as pdbfh:
            self.theory.mm_theory.openmm.app.PDBFile.writeFile(self.theory.mm_theory.topology, newpos, file=pdbfh)

    # Read_data and copydir not used (dummy variables)
    def calc(
        self,
        coords: np.ndarray,
        tmp: StrPath,
        read_data: bool | None = None,
        copydir: StrPath | None = None,
    ) -> CalculationResult:
        if self.iteration_count == self.maxiter:
            raise OpenMMQMMMError(
                f"Geometry optimization stopped: maxiter ({self.maxiter}) reached without convergence"
            )

        # Note: tmp and read_data not used. Needed for geomeTRIC version compatibility
        logger.info("Convergence criteria: %s", self.conv_criteria)

        # Need to combine with rest of full-system coords
        self.M.xyzs[0] = coords.reshape(-1, 3) * openmmqmmm.constants.BOHR_TO_ANG
        currcoords = self.M.xyzs[0]

        if self.active_region is True:
            egdict = self.actregion_calc(currcoords)
        elif self.pbc_active is True:
            logger.debug("Doing PBC opt-step")
            egdict = self.pbc_calc(currcoords)
        else:
            egdict = self.regular_calc(currcoords)

        return egdict

    def actregion_calc(self, currcoords: np.ndarray) -> CalculationResult:
        # Special act-region (for QM/MM) since GeomeTRIC does not handle huge system and constraints
        # The only caller already gates on this; kept as a guard so the method is safe alone.
        if self.active_region is not True:
            raise InternalError("actregion_calc called without an active region")

        full_coords = self.fragment.coords

        for act_i, curr_i in zip(self.actatoms, currcoords, strict=False):
            full_coords[act_i] = curr_i
        self.full_current_coords = full_coords

        # Write out fragment with updated coordinates for the purpose of doing restart
        self.fragment.replace_coords(self.fragment.elems, self.full_current_coords, conn=False)
        self.fragment.print_system(filename="fragment_currentgeo.frag")
        self.fragment.write_xyzfile(xyzfilename="Fragment-currentgeo.xyz")

        logger.info(f"Current geometry (Å) in step {self.iteration_count} (print_atoms_list region)")
        logger.info("-------------------------------------------------")

        print_coords_for_atoms(self.full_current_coords, self.fragment.elems, self.print_atoms_list)
        logger.info("Note: Only print_atoms_list region printed above")

        E, grad = self.theory.run(
            current_coords=self.full_current_coords,
            elems=self.fragment.elems,
            charge=self.charge,
            mult=self.mult,
            grad=True,
        )

        if logger.isEnabledFor(logging.DEBUG):
            write_coords_all(
                grad,
                self.fragment.elems,
                indices=self.fragment.allatoms,
                file="Grad",
                description="Grad (au/Bohr):",
            )
        # Trim Full gradient down to only act-atoms gradient
        Grad_act = np.array([grad[i] for i in self.actatoms])
        if logger.isEnabledFor(logging.DEBUG):
            act_elems = [self.fragment.elems[i] for i in self.actatoms]
            write_coords_all(
                Grad_act,
                act_elems,
                indices=list(range(len(self.actatoms))),
                file="Grad_act",
                description="Grad_act (au/Bohr):",
            )
        self.energy = E

        logger.info("Writing trajectory for Active Region to file: geometric_OPTtraj.xyz")

        self.write_trajectory_full()

        if isinstance(self.theory, QMMMTheory):
            self.write_trajectory_qmregion()
            self.write_energy_logfile()

            if isinstance(self.theory.mm_theory, OpenMMTheory) and self.mm_pdb_traj_write is True:
                self.write_pdbtrajectory()

        step_lines = pygrep2("Step ", "geometric_OPTtraj.log", print_output=False, errors=None)
        if len(step_lines) > 0:
            iteration = int(step_lines[-1].split("Step", 1)[1].split(":", 1)[0].strip())
            self.iteration_count = int(iteration)

        return {"energy": E, "gradient": Grad_act.flatten()}

    def regular_calc(self, currcoords: np.ndarray) -> CalculationResult:
        self.full_current_coords = currcoords
        self.fragment.replace_coords(self.fragment.elems, self.full_current_coords, conn=False)
        self.fragment.write_xyzfile(xyzfilename="Fragment-currentgeo.xyz")
        logger.info(f"Current geometry (Å) in step {self.iteration_count} (print_atoms_list region)")
        logger.info("---------------------------------------------------")
        print_coords_for_atoms(currcoords, self.fragment.elems, self.print_atoms_list)
        logger.info("\nNote: printed only print_atoms_list (this is not necessarily all atoms) ")
        E, grad = self.theory.run(
            current_coords=currcoords, elems=self.M.elem, charge=self.charge, mult=self.mult, grad=True
        )
        step_lines = pygrep2("Step ", "geometric_OPTtraj.log", print_output=False, errors=None)
        if len(step_lines) > 0:
            iteration = int(step_lines[-1].split("Step", 1)[1].split(":", 1)[0].strip())
            self.iteration_count = int(iteration)
        self.energy = E
        return {"energy": E, "gradient": grad.flatten()}

    def pbc_calc(self, currcoords: np.ndarray) -> CalculationResult:
        # Split  coords into atomic and lattic
        R_geo = currcoords[:-4]
        origin = currcoords[-4]
        H_geo = currcoords[-3:] - origin

        logger.info("Enforcing orientation")
        # 1. Ensure the Origin dummy atom stays at exactly 0,0,0
        origin[:] = 0.0
        # 2. Force H_geo to be strictly upper-triangular
        # Vector A: Only Ax is allowed (Ay and Az are zero)
        H_geo[0, 1] = 0.0  # ay = 0
        H_geo[0, 2] = 0.0  # az = 0
        # Vector B: Only Bx and By are allowed (Bz is zero)
        H_geo[1, 2] = 0.0  # bz = 0
        s = np.dot(R_geo - origin, self.H_ref_inv)
        R_phys = np.dot(s, H_geo) + origin
        self.theory.update_cell(H_geo)

        self.full_current_coords = R_phys
        self.fragment.replace_coords(self.fragment.elems, self.full_current_coords, conn=False)
        self.fragment.write_xyzfile(xyzfilename="Fragment-currentgeo.xyz")
        logger.info(f"Current geometry (Å) in step {self.iteration_count} (print_atoms_list region)")
        logger.info("---------------------------------------------------")
        print_coords_for_atoms(R_phys, self.elems_phys, self.print_atoms_list)
        logger.info("\nNote: printed only print_atoms_list (this is not necessarily all atoms) ")
        logger.info(f"Current cell vectors (Å):{H_geo}")
        logger.info(f"Current cell volume (Å³):{cell_volume(H_geo)}")

        E, grad_phys = self.theory.run(
            current_coords=R_phys, elems=self.elems_phys, charge=self.charge, mult=self.mult, grad=True
        )
        self.energy = E

        step_lines = pygrep2("Step ", "geometric_OPTtraj.log", print_output=False, errors=None)
        if len(step_lines) > 0:
            iteration = int(step_lines[-1].split("Step", 1)[1].split(":", 1)[0].strip())
            self.iteration_count = int(iteration)

        # M is the transformation matrix: R_phys = R_geo @ M
        M = np.dot(self.H_ref_inv, H_geo)
        grad_Rgeo = np.dot(grad_phys, M.T)

        # Total lattice gradient: current theory cell-gradient + convection
        grad_latt_total = self.theory.get_cell_gradient()
        # Standard orientation mask:
        # This zeros out: a_y, a_z, and b_z
        mask = np.array(
            [
                [1, 0, 0],  # dE/dax (ay, az frozen)
                [1, 1, 0],  # dE/dbx, dE/dby (bz frozen)
                [1, 1, 1],  # dE/dcx, dE/dcy, dE/dcz (all free)
            ]
        )
        grad_latt_masked = grad_latt_total * mask
        grad_origin = np.zeros((1, 3))
        mod_gradient = np.concatenate(
            [
                grad_Rgeo,  # (N, 3)
                grad_origin,  # (1, 3)
                grad_latt_masked,  # (3, 3)
            ],
            axis=0,
        )

        return {"energy": E, "gradient": mod_gradient.flatten()}


def _constraints_indices_convert(con: ConstraintDict, actatoms: Sequence[int]) -> ConstraintDict:
    try:
        bondcons = con["bond"]
    except KeyError:
        bondcons = []
    try:
        anglecons = con["angle"]
    except KeyError:
        anglecons = []
    try:
        dihedralcons = con["dihedral"]
    except KeyError:
        dihedralcons = []
    for bc in bondcons:
        bc[0] = fullindex_to_actindex(bc[0], actatoms)
        bc[1] = fullindex_to_actindex(bc[1], actatoms)
    for ac in anglecons:
        ac[0] = fullindex_to_actindex(ac[0], actatoms)
        ac[1] = fullindex_to_actindex(ac[1], actatoms)
        ac[2] = fullindex_to_actindex(ac[2], actatoms)
    for dc in dihedralcons:
        dc[0] = fullindex_to_actindex(dc[0], actatoms)
        dc[1] = fullindex_to_actindex(dc[1], actatoms)
        dc[2] = fullindex_to_actindex(dc[2], actatoms)
        dc[3] = fullindex_to_actindex(dc[3], actatoms)
    return con

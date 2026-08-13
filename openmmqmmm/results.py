import logging
from dataclasses import dataclass

import numpy as np

from openmmqmmm.coords import Fragment

logger = logging.getLogger(__name__)


@dataclass
class Results:
    """Container for job results (energies, gradients, frequencies, thermochemistry)."""

    label: str | None = None
    energy: float | None = None
    qm_energy: float | None = None
    mm_energy: float | None = None
    qmmm_energy: float | None = None
    gradient: np.ndarray | None = None
    reaction_energy: float | None = None

    energies: list | None = None
    reaction_energies: list | None = None
    relative_energies: list | None = None
    labels: list | None = None
    gradients: list | None = None
    energies_dict: dict | None = None
    gradients_dict: dict | None = None
    # Name of worker directories that could be accessed later
    worker_dirnames: dict | None = None
    charge: int | None = None
    mult: int | None = None
    properties: dict | None = None
    hessian: np.ndarray | None = None
    frequencies: list | None = None
    freq_masses: list | None = None
    freq_elems: list | None = None
    freq_coords: np.ndarray | None = None
    freq_atoms: list | None = None
    freq_tr_modenum: int | None = None
    freq_projection: bool | None = None
    freq_scaling_factor: float | None = None
    freq_dipole_derivs: np.ndarray | None = None
    freq_polarizability_derivs: np.ndarray | None = None
    freq_raman: bool | None = None
    normal_modes: np.ndarray | None = None
    raman_activities: np.ndarray | None = None
    ir_intensities: np.ndarray | None = None
    depolarization_ratios: np.ndarray | None = None
    vib_eigenvectors: np.ndarray | None = None
    thermochemistry: dict | None = None
    displacement_dipole_dictionary: dict | None = None
    displacement_polarizability_dictionary: dict | None = None

    def write_to_disk(self, filename="results.json"):
        """Write the defined attributes to a JSON file."""
        import json

        logger.info("\nWriting to disk defined attributes of Results dataclass")

        newdict = {}
        for k, v in self.__dict__.items():
            if isinstance(v, np.ndarray):
                if np.any(np.isnan(v)):
                    logger.warning(f"NaN found in array {k}")
                    logger.info("Skipping writing to disk")
                else:
                    newv = v.tolist()
                    newdict[k] = newv
            # Dealing with cases of lists of np arrays (e.g. pol derivs)
            elif isinstance(v, list):
                if len(v) == 0:
                    newdict[k] = v
                elif isinstance(v[0], np.ndarray):
                    newv = [i.tolist() for i in v]
                    newdict[k] = newv
                else:
                    newdict[k] = v
            elif isinstance(v, Fragment):
                logger.warning("Fragment objects are not included in the results file on disk")
            else:
                newdict[k] = v

        logger.info("Results object data:")
        for k, v in newdict.items():
            if type(v) is list or type(v) is np.ndarray:
                if len(v) < 20:
                    logger.info(f"{k} : {len(v)}")
                else:
                    logger.info(f"{k} : too long to print")
            elif v is not None:
                logger.info(f"{k} : {v}")
        try:
            with open(filename, "w") as f:
                f.write(json.dumps(newdict, allow_nan=True))
        except TypeError as e:
            logger.error(f"Failed to write Results to disk: {e}")
            logger.info("Skipping writing to disk")
            return


def read_results_from_file(filename="results.json") -> "Results":
    """Read a Results object from a JSON file written by Results.write_to_disk."""
    import json
    from dataclasses import fields

    logger.info("Reading Results data from file:")
    with open(filename) as f:
        data = json.load(f)
    logger.info("Data read from file:")
    for k, v in data.items():
        logger.info(f"{k} : {v}")

    # Ignore keys from files written by older versions with more fields
    known_fields = {f.name for f in fields(Results)}
    return Results(**{k: v for k, v in data.items() if k in known_fields})

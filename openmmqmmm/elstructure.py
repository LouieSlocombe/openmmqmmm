"""Electronic-structure analysis helpers (electronic entropy, XDM dispersion via postg, CM5 charges)."""

import logging
import os
import shutil
import subprocess as sp

import numpy as np

import openmmqmmm.coords
from openmmqmmm.elements import cm5_dz, cm5_radii
from openmmqmmm.exceptions import ExternalProgramError, InternalError

logger = logging.getLogger(__name__)

# Remaining CM5 parameters from the paper; cm5_radii and cm5_dz are the per-element columns.
_alpha = 2.474
_DHC = 0.0502
_DHN = 0.1747
_DHO = 0.1671
_DCN = 0.0556
_DCO = 0.0234
_DNO = -0.0346


# Get list-of-lists of distances of coords
def distance_matrix_from_coords(coords):
    distmatrix = []
    for i in coords:
        dist_row = [openmmqmmm.coords.distance(i, j) for j in coords]
        distmatrix.append(dist_row)
    return distmatrix


def calc_cm5(atomic_numbers, coords, hirschfeldcharges):
    coords = np.array(coords)
    atomic_numbers = np.array(atomic_numbers)
    # all matrices have the naming scheme matrix[k,k'] according to the paper
    distances = np.array(distance_matrix_from_coords(coords))
    Rz = cm5_radii[atomic_numbers - 1]
    RzSum = np.tile(Rz, (len(Rz), 1))
    RzSum = np.add(RzSum, np.transpose(RzSum))
    Bkk = np.exp(-_alpha * (np.subtract(distances, RzSum)), out=np.zeros_like(distances), where=distances != 0)
    if not (np.diagonal(Bkk) == 0).all():
        raise InternalError("Bkk matrix diagonal is not zero")

    Tkk = np.zeros(shape=Bkk.shape)
    shape = Tkk.shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            numbers = [atomic_numbers[i], atomic_numbers[j]]
            if numbers[0] == numbers[1]:
                continue
            if set(numbers) == {1, 6}:
                Tkk[i, j] = _DHC
                if numbers == [6, 1]:
                    Tkk[i, j] *= -1.0
            elif set(numbers) == {1, 7}:
                Tkk[i, j] = _DHN
                if numbers == [7, 1]:
                    Tkk[i, j] *= -1.0
            elif set(numbers) == {1, 8}:
                Tkk[i, j] = _DHO
                if numbers == [8, 1]:
                    Tkk[i, j] *= -1.0
            elif set(numbers) == {6, 7}:
                Tkk[i, j] = _DCN
                if numbers == [7, 6]:
                    Tkk[i, j] *= -1.0
            elif set(numbers) == {6, 8}:
                Tkk[i, j] = _DCO
                if numbers == [8, 6]:
                    Tkk[i, j] *= -1.0
            elif set(numbers) == {7, 8}:
                Tkk[i, j] = _DNO
                if numbers == [8, 7]:
                    Tkk[i, j] *= -1.0
            else:
                Tkk[i, j] = cm5_dz[numbers[0] - 1] - cm5_dz[numbers[1] - 1]
    if not (np.diagonal(Tkk) == 0).all():
        raise InternalError("Tkk matrix diagonal is not zero")
    product = np.multiply(Tkk, Bkk)
    if not (np.diagonal(product) == 0).all():
        raise InternalError("Product matrix diagonal is not zero")
    result = np.sum(product, axis=1)
    return np.array(hirschfeldcharges) + result


# Interface to XDM postg program
# https://github.com/aoterodelaroza/postg
def xdm_run(wfxfile=None, postgdir=None, a1=None, a2=None, functional=None):
    if postgdir is None:
        # Trying to find postgdir in path
        logger.info("postgdir keyword argument not provided to xdm_run. Trying to find postg in PATH")
        try:
            postgdir = os.path.dirname(shutil.which("postg"))
            logger.info("Found postg in path. Setting postgdir.")
        except TypeError:
            raise ExternalProgramError("Found no postg executable in PATH") from None

    parameterdict = {
        "pw86pbe": [0.7564, 1.4545],
        "b3lyp": [0.6356, 1.5119],
        "b3pw91": [0.6002, 1.4043],
        "b3p86": [1.0400, 0.3741],
        "pbe0": [0.4186, 2.6791],
        "camb3lyp": [0.3248, 2.8607],
        "b97-1": [0.1998, 3.5367],
        "bhandhlyp": [0.5610, 1.9894],
        "blyp": [0.7647, 0.8457],
        "pbe": [0.4492, 2.5517],
        "lcwpbe": [1.0149, 0.6755],
        "tpss": [0.6612, 1.5111],
        "b86bpbe": [0.7443, 1.4072],
    }

    if a1 is None or a2 is None:
        logger.info("a1/a2 parameters not given. Looking up functional in table")
        logger.info("Parameter table: %s", parameterdict)
        a1, a2 = parameterdict[functional.lower()]
        logger.info(f"XDM a1: {a1}, a2: {a2}")
    with open("xdm-postg.out", "w") as ofile:
        sp.run(
            [postgdir + "/postg", str(a1), str(a2), str(wfxfile), str(functional)],
            check=True,
            stdout=ofile,
            stderr=ofile,
            text=True,
        )

    dispgrab = False
    dispgradient = []
    with open("xdm-postg.out") as xdmfile:
        for line in xdmfile:
            if "dispersion energy" in line:
                dispenergy = float(line.split()[-1])
            if "dispersion force constant matrix" in line:
                dispgrab = False
            if dispgrab and "#" not in line:
                grad_x = -1 * float(line.split()[1])
                grad_y = -1 * float(line.split()[2])
                grad_z = -1 * float(line.split()[3])
                dispgradient.append([grad_x, grad_y, grad_z])
            if "dispersion forces" in line:
                dispgrab = True

    dispgradient = np.array(dispgradient)
    logger.info("dispenergy: %s", dispenergy)
    logger.info("dispgradient: %s", dispgradient)
    return dispenergy, dispgradient


# Get electron correlation energy as a function of occupation numbers, sigma and the chosen distribution
def get_ec_entropy(occ, sigma, method="fermi", alpha=0.6):
    from scipy.special import erfinv

    f = occ / 2.0
    f = f[(f > 0) & (f < 1)]
    mask = f > 0.5
    f[mask] = 1.0 - f[mask]
    if method == "fermi":
        fc = f * np.log(f) + (1 - f) * np.log(1 - f)
    elif method == "gaussian":
        fc = -np.exp(-((erfinv(1 - f * 2)) ** 2)) / 2.0 / np.sqrt(np.pi)
    elif method == "linear":
        fc = -f + np.sqrt(2) * f ** (3.0 / 2.0) * 2.0 / 3.0
    else:
        raise ValueError("Not support", method)
    return 2.0 * sigma * fc.sum()

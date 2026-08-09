import logging
import os
import shutil
import subprocess as sp

import numpy as np

import openmmqmmm.coords
from openmmqmmm.exceptions import ExternalProgramError, InternalError

logger = logging.getLogger(__name__)

# CM5 parameters (data from paper for elements 1-118)
_radii = np.array(
    [
        0.32,
        0.37,
        1.30,
        0.99,
        0.84,
        0.75,
        0.71,
        0.64,
        0.60,
        0.62,
        1.60,
        1.40,
        1.24,
        1.14,
        1.09,
        1.04,
        1.00,
        1.01,
        2.00,
        1.74,
        1.59,
        1.48,
        1.44,
        1.30,
        1.29,
        1.24,
        1.18,
        1.17,
        1.22,
        1.20,
        1.23,
        1.20,
        1.20,
        1.18,
        1.17,
        1.16,
        2.15,
        1.90,
        1.76,
        1.64,
        1.56,
        1.46,
        1.38,
        1.36,
        1.34,
        1.30,
        1.36,
        1.40,
        1.42,
        1.40,
        1.40,
        1.37,
        1.36,
        1.36,
        2.38,
        2.06,
        1.94,
        1.84,
        1.90,
        1.88,
        1.86,
        1.85,
        1.83,
        1.82,
        1.81,
        1.80,
        1.79,
        1.77,
        1.77,
        1.78,
        1.74,
        1.64,
        1.58,
        1.50,
        1.41,
        1.36,
        1.32,
        1.30,
        1.30,
        1.32,
        1.44,
        1.45,
        1.50,
        1.42,
        1.48,
        1.46,
        2.42,
        2.11,
        2.01,
        1.90,
        1.84,
        1.83,
        1.80,
        1.80,
        1.73,
        1.68,
        1.68,
        1.68,
        1.65,
        1.67,
        1.73,
        1.76,
        1.61,
        1.57,
        1.49,
        1.43,
        1.41,
        1.34,
        1.29,
        1.28,
        1.21,
        1.22,
        1.36,
        1.43,
        1.62,
        1.75,
        1.65,
        1.57,
    ]
)

_Dz = np.array(
    [
        0.0056,
        -0.1543,
        0.0000,
        0.0333,
        -0.1030,
        -0.0446,
        -0.1072,
        -0.0802,
        -0.0629,
        -0.1088,
        0.0184,
        0.0000,
        -0.0726,
        -0.0790,
        -0.0756,
        -0.0565,
        -0.0444,
        -0.0767,
        0.0130,
        0.0000,
        0.0000,
        0.0000,
        0.0000,
        0.0000,
        0.0000,
        0.0000,
        0.0000,
        0.0000,
        0.0000,
        0.0000,
        -0.0512,
        -0.0557,
        -0.0533,
        -0.0399,
        -0.0313,
        -0.0541,
        0.0092,
        0.0000,
        0.0000,
        0.0000,
        0.0000,
        0.0000,
        0.0000,
        0.0000,
        0.0000,
        0.0000,
        0.0000,
        0.0000,
        -0.0361,
        -0.0393,
        -0.0376,
        -0.0281,
        -0.0220,
        -0.0381,
        0.0065,
        0.0000,
        0.0000,
        0.0000,
        0.0000,
        0.0000,
        0.0000,
        0.0000,
        0.0000,
        0.0000,
        0.0000,
        0.0000,
        0.0000,
        0.0000,
        0.0000,
        0.0000,
        0.0000,
        0.0000,
        0.0000,
        0.0000,
        0.0000,
        0.0000,
        0.0000,
        0.0000,
        0.0000,
        0.0000,
        -0.0255,
        -0.0277,
        -0.0265,
        -0.0198,
        -0.0155,
        -0.0269,
        0.0046,
        0.0000,
        0.0000,
        0.0000,
        0.0000,
        0.0000,
        0.0000,
        0.0000,
        0.0000,
        0.0000,
        0.0000,
        0.0000,
        0.0000,
        0.0000,
        0.0000,
        0.0000,
        0.0000,
        0.0000,
        0.0000,
        0.0000,
        0.0000,
        0.0000,
        0.0000,
        0.0000,
        0.0000,
        0.0000,
        -0.0179,
        -0.0195,
        -0.0187,
        -0.0140,
        -0.0110,
        -0.0189,
    ]
)

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
    Rz = _radii[atomic_numbers - 1]
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
                Tkk[i, j] = _Dz[numbers[0] - 1] - _Dz[numbers[1] - 1]
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
    Ec = 2.0 * sigma * fc.sum()
    return Ec

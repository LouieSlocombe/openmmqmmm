"""Tests for the ORCA output parsers.

These are pure text -> number functions and they had no coverage at all, even though
they are the layer that breaks silently when ORCA changes its output formatting: a
parser that stops matching returns an empty result rather than an error. Writing
these tests found exactly that — grab_orca_timings matched one of its nine labels
against ORCA 6.1.1 output, including the point-charge gradient timing that the QM/MM
path reports.

The fixtures in orca_outputs/ are real ORCA 6.1.1 output for a water molecule
(HF/def2-SVP gradient, BP86/def2-SVP gradient with a point-charge field, and an
HF/def2-SVP frequency run), with only the banner trimmed off the top. The inputs that
produced them are committed alongside; see orca_outputs/README.md to regenerate.
"""

import numpy as np
import pytest

from openmmqmmm import orca
from openmmqmmm.exceptions import FileFormatError

# Reference values are the ones ORCA itself printed in these outputs.
ENGRAD_ENERGY = -75.960983986705
ENGRAD_GRADIENT = [
    [0.0, 0.0, 0.018044520393],
    [-0.0, 0.011070335192, -0.009022260196],
    [-0.0, -0.011070335192, -0.009022260196],
]


@pytest.fixture
def orca_outputs(request):
    """Directory holding the committed ORCA reference outputs."""
    return request.path.parent / "orca_outputs"


def test_grab_final_energy(orca_outputs):
    assert orca.grab_orca_final_energy(str(orca_outputs / "h2o_engrad.out")) == ENGRAD_ENERGY


def test_grab_final_energy_missing_returns_none(tmp_path):
    """A truncated or failed run must report no energy rather than raise."""
    empty = tmp_path / "empty.out"
    empty.write_text("ORCA started but produced nothing useful\n")
    assert orca.grab_orca_final_energy(str(empty)) is None


def test_check_orca_finished(orca_outputs, tmp_path):
    finished, scf_iterations = orca.check_orca_finished(str(orca_outputs / "h2o_engrad.out"))
    assert finished is True
    assert scf_iterations == "11"

    truncated = tmp_path / "truncated.out"
    truncated.write_text("SCF CONVERGED AFTER  11 CYCLES\n")
    assert orca.check_orca_finished(str(truncated)) == (False, None)


def test_grab_gradient(orca_outputs):
    gradient = orca.grab_orca_gradient(str(orca_outputs / "h2o_engrad.engrad"))
    assert gradient.shape == (3, 3)
    assert np.allclose(gradient, ENGRAD_GRADIENT)
    # Water is symmetric about the C2 axis: the two hydrogens mirror each other in y.
    assert gradient[1, 1] == pytest.approx(-gradient[2, 1])


def test_grab_pc_gradient(orca_outputs):
    pc_gradient = orca.grab_orca_pc_gradient(str(orca_outputs / "h2o_pc.pcgrad"))
    assert pc_gradient.shape == (2, 3), "One row per point charge"
    # The two charges sit in the yz-plane, so the x-component is exactly zero.
    assert np.allclose(pc_gradient[0], [0.0, 0.005690336178, 0.003780363794])


@pytest.mark.parametrize(
    ("chargemodel", "expected"),
    [("mulliken", [-0.347486, 0.173743, 0.173743]), ("loewdin", [-0.156043, 0.078021, 0.078021])],
)
def test_grab_atom_charges(orca_outputs, chargemodel, expected):
    charges = orca.grab_orca_atom_charges(chargemodel, str(orca_outputs / "h2o_engrad.out"))
    assert charges == pytest.approx(expected)
    # ORCA prints six decimals, so the printed charges sum to the total only to that precision.
    assert sum(charges) == pytest.approx(0.0, abs=1e-5), "Charges must sum to the total charge"


def test_grab_dipole_moment(orca_outputs):
    dipole = orca.grab_dipole_moment(str(orca_outputs / "h2o_engrad.out"))
    assert len(dipole) == 3
    # Water's dipole lies along the C2 (z) axis.
    assert dipole[0] == pytest.approx(0.0, abs=1e-9)
    assert dipole[1] == pytest.approx(0.0, abs=1e-9)
    assert dipole[2] == pytest.approx(-0.839370865)


def test_grab_timings(orca_outputs):
    """The timing labels must survive ORCA's varying column widths.

    The previous implementation matched fixed-width strings such as
    "Sum of individual times         ...:" and found only one label in this output.
    """
    timings = orca.grab_orca_timings(str(orca_outputs / "h2o_engrad.out"))
    assert "total_time" in timings
    assert "time_scfiterations" in timings
    assert "time_scfgrad" in timings
    assert all(value >= 0.0 for value in timings.values())


def test_grab_timings_finds_pointcharge_gradient(orca_outputs):
    """pc_gradient is the one timing the QM/MM path actually reports."""
    timings = orca.grab_orca_timings(str(orca_outputs / "h2o_pc.out"))
    assert "pc_gradient" in timings
    assert "rij_coulomb_gradient" in timings
    assert "xc_gradient" in timings


def test_grab_timings_unreadable_file_is_empty(tmp_path):
    assert orca.grab_orca_timings(str(tmp_path / "does_not_exist.out")) == {}


def test_grab_hessian(orca_outputs):
    hessian = orca.grab_hessian(str(orca_outputs / "h2o_freq.hess"))
    assert hessian.shape == (9, 9), "3N x 3N for three atoms"
    assert np.allclose(hessian, hessian.T), "The Hessian must be symmetric"


def test_hessian_write_read_roundtrip(orca_outputs, tmp_path):
    """write_orca_hessfile must produce a file grab_hessian can read back.

    It did not: the written file has no section after $hessian, so the reader ran on
    into the $atoms block, treated those lines as matrix rows and raised IndexError.
    """
    hessian = orca.grab_hessian(str(orca_outputs / "h2o_freq.hess"))
    elems = ["O", "H", "H"]
    coords = [[0.0, 0.0, 0.1173], [0.0, 0.7572, -0.4692], [0.0, -0.7572, -0.4692]]
    masses = [15.999, 1.008, 1.008]

    outputname = str(tmp_path / "roundtrip.hess")
    orca.write_orca_hessfile(hessian, coords, elems, masses, list(range(3)), outputname)

    assert np.allclose(orca.grab_hessian(outputname), hessian)


def test_grab_ir_intensities(orca_outputs):
    """IR intensities come from the $ir_spectrum block of the .hess file."""
    intensities = orca.grab_ir_intensities(str(orca_outputs / "h2o_freq.hess"))
    assert len(intensities) == 9, "One row per Cartesian degree of freedom (3N)"
    assert all(intensity >= 0.0 for intensity in intensities)
    # Six zero-frequency translations/rotations carry no intensity; three modes remain.
    assert sum(intensity > 0.0 for intensity in intensities) == 3


def test_grab_warnings_reports_real_warnings(tmp_path, caplog):
    """Boilerplate warning banners are filtered out; real warnings are reported."""
    outfile = tmp_path / "warned.out"
    outfile.write_text(
        "                                        WARNINGS\n"
        "WARNING: Old DensityContainer found\n"
        "WARNING: geometry is not fully converged\n"
    )
    with caplog.at_level("INFO", logger="openmmqmmm.orca"):
        orca.grab_orca_warnings(str(outfile))

    assert "geometry is not fully converged" in caplog.text
    assert "Old DensityContainer" not in caplog.text, "Known-benign warnings are filtered"


def test_clean_output_has_no_errors(orca_outputs, caplog):
    """A successful run must not be reported as having errors."""
    with caplog.at_level("INFO", logger="openmmqmmm.orca"):
        orca.grab_orca_errors(str(orca_outputs / "h2o_engrad.out"))
    assert "ORCA-error" not in caplog.text


# The open-shell fixture (water cation, UHF/def2-SVP) carries all four charge tables the
# parsers know how to read plus the "... AND SPIN POPULATIONS" variants, so the table
# scanner behind grab_orca_atom_charges is covered for every model it claims to support.
CATION_CHARGES = {
    "Mulliken": [0.381621, 0.309190, 0.309190],
    "Loewdin": [0.608679, 0.195660, 0.195660],
    "CHELPG": [-0.011262, 0.505539, 0.505723],
    "Hirshfeld": [0.336715, 0.331644, 0.331644],
}
CATION_SPIN_POPULATIONS = {
    "Mulliken": [1.092995, -0.046497, -0.046497],
    "Loewdin": [1.013492, -0.006746, -0.006746],
}


@pytest.fixture
def cation_output(orca_outputs):
    return str(orca_outputs / "h2o_cation_charges.out")


@pytest.mark.parametrize(("chargemodel", "expected"), CATION_CHARGES.items())
def test_grab_atom_charges_every_model(cation_output, chargemodel, expected):
    charges = orca.grab_orca_atom_charges(chargemodel, cation_output)
    assert charges == pytest.approx(expected)
    # The cation carries a single positive charge.
    assert sum(charges) == pytest.approx(1.0, abs=1e-5)


@pytest.mark.parametrize("chargemodel", list(CATION_CHARGES))
def test_chargemodel_name_is_case_insensitive(cation_output, chargemodel):
    """Callers pass these straight through from user input, in whatever case they wrote."""
    assert orca.grab_orca_atom_charges(chargemodel.upper(), cation_output) == pytest.approx(
        orca.grab_orca_atom_charges(chargemodel.lower(), cation_output)
    )


def test_grab_atom_charges_rejects_unknown_model(cation_output):
    with pytest.raises(FileFormatError):
        orca.grab_orca_atom_charges("NotAChargeModel", cation_output)


@pytest.mark.parametrize(("chargemodel", "expected"), CATION_SPIN_POPULATIONS.items())
def test_grab_spin_populations(cation_output, chargemodel, expected):
    spinpops = orca.grab_orca_spin_populations(chargemodel, cation_output)
    assert spinpops == pytest.approx(expected)
    # A doublet has one unpaired electron.
    assert sum(spinpops) == pytest.approx(1.0, abs=1e-5)


def test_charge_tables_are_not_confused_with_spin_tables(cation_output):
    """The plain-charge and charge-plus-spin tables share a heading prefix.

    "MULLIKEN ATOMIC CHARGES" is a prefix of "MULLIKEN ATOMIC CHARGES AND SPIN
    POPULATIONS", so the charge parser matches the open-shell table too and has to take
    the second-to-last column there rather than the last one. Getting that wrong returns
    spin populations while claiming to return charges.
    """
    charges = orca.grab_orca_atom_charges("Mulliken", cation_output)
    spinpops = orca.grab_orca_spin_populations("Mulliken", cation_output)
    assert charges != pytest.approx(spinpops)
    assert charges == pytest.approx(CATION_CHARGES["Mulliken"])


def test_grab_spin_populations_rejects_unknown_model(cation_output):
    with pytest.raises(FileFormatError):
        orca.grab_orca_spin_populations("NotAChargeModel", cation_output)


def test_grab_cm5_charges(cation_output):
    """CM5 is derived from the Hirshfeld charges plus the geometry ORCA used for them.

    It is the one charge model that needs a second block out of the output file, and it
    had no coverage at all.
    """
    cm5 = orca.grab_orca_atom_charges("CM5", cation_output)
    assert len(cm5) == 3
    assert sum(cm5) == pytest.approx(1.0, abs=1e-4), "The cation carries one positive charge"
    # CM5 shifts charge from hydrogen towards oxygen relative to Hirshfeld.
    hirshfeld = orca.grab_orca_atom_charges("Hirshfeld", cation_output)
    assert cm5[0] < hirshfeld[0]
    assert cm5[1] > hirshfeld[1]

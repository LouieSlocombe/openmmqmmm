import pytest

from openmmqmmm.exceptions import InputError
from openmmqmmm.mdtraj import mdtraj_image_trajectory


def test_unknown_trajectory_format_is_rejected_before_loading():
    with pytest.raises(InputError, match="XTC"):
        mdtraj_image_trajectory("missing.dcd", "missing.pdb", traj_format="XTC")

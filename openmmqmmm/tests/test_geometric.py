from openmmqmmm import Fragment, ZeroTheory, optimize_geometry


def test_geometric_dummy():
    # Define coordinate string
    coords = """
    O       -1.377626260      0.000000000     -1.740199718
    H       -1.377626260      0.759337000     -1.144156718
    H       -1.377626260     -0.759337000     -1.144156718
    """
    # Defining fragment
    H2Ofragment = Fragment(coordsstring=coords, charge=0, mult=1)

    # Defining dummy theory
    zerotheorycalc = ZeroTheory()

    # Optimize with dummy theory: exercises the geomeTRIC coupling
    result = optimize_geometry(fragment=H2Ofragment, theory=zerotheorycalc)

    assert result.energy == 0.0, "ZeroTheory energy should be 0.0"

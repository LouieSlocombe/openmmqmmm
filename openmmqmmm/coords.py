"""Fragment and Reaction classes plus coordinate and topology utilities.

Covers XYZ/PDB/Amber/GROMACS I/O, connectivity, alignment and QM-region tools.
"""

import copy
import logging
import math
import os
import time
from collections import Counter, defaultdict
from math import sqrt

import numpy as np

import openmmqmmm.constants
import openmmqmmm.elements

# Re-exported: the tables live in elements.py, one row per element, but they have been
# importable from coords since before that module existed and callers still use them here.
from openmmqmmm.elements import atommasses, eldict_covrad, elematomnumbers
from openmmqmmm.exceptions import (
    FileFormatError,
    InputError,
    InternalError,
    MissingDependencyError,
)
from openmmqmmm.utils import (
    isint,
    listdiff,
    log_time_since,
    natural_sort,
    search_list_of_lists_for_index,
    small_header,
    sub_header,
    sub_header_end,
)

logger = logging.getLogger(__name__)

# Default connectivity parameters: covalent-radius scaling factor and tolerance
CONNECTIVITY_SCALE = 1.0
CONNECTIVITY_TOL = 0.1


# Reaction class: connects list of fragments and stoichiometry
# TODO: Check that the charge and multiplicity is consistent with formula. Maybe do in fragment instead?
# TODO: Check charge on both sides of reaction. Warning if different.
# TODO: Check if mult is different on both sides of reaction. Print warning

# FUNCTIONS that could interact with Reaction class:
# Singlepoint_reaction ?,
# Optimizer ? Probably not


class Reaction:
    """A reaction: an ordered list of fragments with stoichiometry (and optional energies)."""

    def __init__(self, fragments, stoichiometry, label=None, unit="eV"):
        logger.info(sub_header("New reaction"))

        # Reading fragments and checking for charge/mult and matching stoichiometry
        self.fragments = fragments
        self.stoichiometry = stoichiometry
        self.check_fragments()
        # List of all elements in reaction
        self.elements = [item for sublist in [frag.elems for frag in fragments] for item in sublist]

        self.label = label

        self.unit = unit

        # List of energies for each fragment
        self.energies = []
        # Reaction energy
        self.reaction_energy = None

        # Keeping track of orbital-files: key: 'SCF':["frag1.gbw","frag2.gbw","frag3.gbw"],
        # 'MP2nat':["frag1.gbw","frag2.gbw","frag3.gbw"]
        self.orbital_dictionary = defaultdict(list)
        # Keep track of various properties calculated
        self.properties = defaultdict(list)

    def reset_energies(self):
        """Discard the stored fragment energies and reaction energy."""
        # Reset energies etc
        self.energies = []
        self.reaction_energy = None

    def check_fragments(self):
        """Validate the reaction definition.

        Raises:
            InputError: if a fragment is missing charge/multiplicity, or if there is
                not exactly one stoichiometry coefficient per fragment.
        """
        for frag in self.fragments:
            if frag.charge is None or frag.mult is None:
                raise InputError(f"Error: Missing charge/mult information in fragment: {frag.formula}")
        # Checked here rather than in reaction_energy so a mis-typed stoichiometry fails
        # immediately instead of after every fragment has been through the QM program.
        if len(self.stoichiometry) != len(self.fragments):
            raise InputError(
                f"Error: {len(self.stoichiometry)} stoichiometry values for "
                f"{len(self.fragments)} fragments. One signed coefficient per fragment is required."
            )

    def calculate_reaction_energy(self):
        """Combine the stored fragment energies into the reaction energy.

        Sets self.reaction_energy, in self.unit. Logs a warning and does nothing if
        energies are missing for some fragments.
        """
        if len(self.energies) == len(self.fragments):
            self.reaction_energy = openmmqmmm.reaction_energy(
                list_of_energies=self.energies,
                stoichiometry=self.stoichiometry,
                unit=self.unit,
                silent=False,
                label=self.label,
            )[0]
        else:
            logger.warning("Could not calculate reaction energy as we are missing energies for fragments")


# Fragment class
class Fragment:
    """Molecular system: elements, coordinates, charge/multiplicity, connectivity and topology.

    Create from a coordinate string (coordsstring=), lists (elems=/coords=), an XYZ file
    (xyzfile=), a PDB file (pdbfile=), Amber/GROMACS/chemshell files, or a fragment file
    written by print_system (fragfile=).
    """

    def __init__(
        self,
        fragments=None,
        coordsstring=None,
        fragfile=None,
        xyzfile=None,
        pdbfile=None,
        pdbxfile=None,
        grofile=None,
        amber_inpcrdfile=None,
        amber_prmtopfile=None,
        smiles=None,
        chemshellfile=None,
        coords=None,
        elems=None,
        connectivity=None,
        atom=None,
        diatomic=None,
        diatomic_bondlength=None,
        bondlength=None,
        atomcharges=None,
        atomtypes=None,
        conncalc=False,
        scale=None,
        tol=None,
        charge=None,
        mult=None,
        label=None,
        readchargemult=False,
        use_atomnames_as_elements=False,
    ):

        # Defining initial charge/mult attributes. Will be redefined
        self.charge = None
        self.mult = None

        # Setting initial dummy label. Possibly redefined below, either when reading in file or by label keyword
        self.label = None

        # Printlevel. Default: 2 (slightly verbose)

        logger.info(sub_header("New fragment"))
        # Minimal Fragment
        logger.info("Fragment creation")
        self.energy = None
        self.elems = []
        self.coords = np.zeros((0, 3))
        self.connectivity = []
        self.atomcharges = []
        self.atomtypes = []
        # Optional PDB-information that might be stored (if PDB-file read)
        self.pdb_atomnames = None
        self.pdb_resnames = None
        self.pdb_chainlabels = None
        self.pdb_residlabels = None
        self.pdb_conect_lines = None
        self.pdb_topology = None  # New, use OpenMM to read PDB-file and get topology
        # Atomnames in a forcefield sense
        self.Centralmainfrag = []
        self.formula = None
        if atomcharges is not None:
            self.atomcharges = atomcharges
        if atomtypes is not None:
            self.atomtypes = atomtypes
        # if atomnames is not None:
        # Hessian. Can be added by Numfreq/Anfreq job
        self.hessian = None

        # Needed for print_system
        # Todo: revisit this
        self.fragmenttype_labels = []

        # Here either providing coords, elems as lists.
        ##############################
        # NOW PROCESSING INPUT DATA
        ##############################
        # Lists of elements and coordinates provided
        if coords is not None:
            # Adding coords as list of lists (or np.array). Conversion to numpy array
            self.coords = reformat_list_to_array(coords)
            if elems is None:
                raise InputError("Error: Coords list provided but no elems list. Exiting.")
            if len(elems) != len(coords):
                raise InputError(
                    f"Error: Coords list (len {len(coords)}) and elems list ({len(elems)}) have different lengths. "
                    f"Exiting."
                )
            self.elems = elems
            # If connectivity passed
            if connectivity is not None:
                conncalc = False
                self.connectivity = connectivity

        # Fragment from input fragments
        elif fragments is not None:
            logger.info("Creating fragments by combining input fragments")
            self.elems = []
            for f in fragments:
                self.elems += f.elems
            self.coords = np.vstack([f.coords for f in fragments])

            # Use charge/mult if provided, otherwise use
            if charge is None:
                logger.info("Combining charge and multiplicities from input fragments")
                try:
                    charges_fragments = [f.charge for f in fragments]
                    charge = sum(charges_fragments)
                    mults_fragments = [f.mult for f in fragments]
                    spin_fragments = [(m - 1) / 2 for m in mults_fragments]
                    spin = sum(spin_fragments)
                    mult = int(2 * spin + 1)
                except TypeError:
                    logger.info("Charges/multiplicities not found in inputfragments.")

        # Defining an atom
        elif atom is not None:
            logger.info("Creating Atom Fragment")
            self.elems = [atom]
            self.coords = reformat_list_to_array([[0.0, 0.0, 0.0]])
        # Defining a diatomic
        elif diatomic is not None:
            logger.info("Creating Diatomic Fragment from formula and bondlength")
            if bondlength is None:
                # TODO: remove diatomic_bondlength and use bondlength only
                if diatomic_bondlength is None:
                    raise InputError("diatomic option requires bondlength to be set. Exiting!")
                bondlength = diatomic_bondlength
            self.elems = molformulatolist(diatomic)
            if len(self.elems) != 2:
                raise InputError(f"Problem with molecular formula diatomic={diatomic} string!")
            self.coords = reformat_list_to_array([[0.0, 0.0, 0.0], [0.0, 0.0, float(bondlength)]])
        # If coordsstring given, read elems and coords from it
        elif coordsstring is not None:
            self.add_coords_from_string(coordsstring, scale=scale, tol=tol, conncalc=conncalc)
        # If smiles given, create coords from it
        elif smiles is not None:
            self.create_coords_from_smiles(smiles)
        # If xyzfile argument, run read_xyzfile
        elif xyzfile is not None:
            if not os.path.isfile(xyzfile):
                raise InputError(f"XYZ-file {xyzfile} not found. Exiting.")

            self.label = xyzfile.split("/")[-1].split(".")[0]
            self.read_xyzfile(xyzfile, readchargemult=readchargemult, conncalc=conncalc)
        # PDB-file
        elif pdbfile is not None:
            self.label = pdbfile.split("/")[-1].split(".")[0]
            self.read_pdbfile_openmm(pdbfile)
        # PDBX-file
        elif pdbxfile is not None:
            self.label = pdbxfile.split("/")[-1].split(".")[0]
            self.read_pdbxfile(pdbxfile)
        # GROMACS GRO-file
        elif grofile is not None:
            self.label = grofile.split("/")[-1].split(".")[0]
            self.read_grofile(grofile, conncalc=False)
        # Amber CRD file (requires prmtop file as well)
        elif amber_inpcrdfile is not None:
            self.label = amber_inpcrdfile.split("/")[-1].split(".")[0]
            logger.info("Reading Amber INPCRD file")
            if amber_prmtopfile is None:
                raise InputError("amber_prmtopfile argument must be provided as well!")
            self.read_amberfile(inpcrdfile=amber_inpcrdfile, prmtopfile=amber_prmtopfile, conncalc=conncalc)
        elif chemshellfile is not None:
            self.label = chemshellfile.split("/")[-1].split(".")[0]
            self.read_chemshellfile(chemshellfile, conncalc=conncalc)
        # fragment file
        elif fragfile is not None:
            self.label = fragfile.split("/")[-1].split(".")[0]
            self.read_fragment_from_file(fragfile)
        # If all else fails, exit
        else:
            raise InputError("Fragment requires some kind of valid coordinate input!")
        # Label for fragment (string). Useful for distinguishing different fragments
        # This overrides label-definitions above (self.label=xyzfile etc)
        if label is not None:
            self.label = label

        # Now set charge and mult attributes of fragment from keyword arg unless None. Will override readchargemult
        # option above if used
        if charge is not None:
            self.charge = charge
        if mult is not None:
            self.mult = mult

        # Now update attributes after defining coordinates, getting charge, mult
        self.update_attributes()
        if conncalc is True and len(self.connectivity) == 0:
            self.calc_connectivity(scale=scale, tol=tol)

        # Constraints attributes. Used by parallel surface-scan to pass constraints along.
        # Populated by calc_surface relaxed para
        self.constraints = None

    def __repr__(self):
        logger.info("Fragment object")
        logger.info(f"Number of Atoms in fragment: {self.numatoms}")
        logger.info(f"Formula: {self.prettyformula}")
        logger.info(f"Label: {self.label}")
        logger.info(f"Charge: {self.charge} Mult: {self.mult}")
        logger.info("Do fragment.info() for more info on fragment")
        return "fragment"

    def __str__(self):
        logger.info("Fragment object")
        logger.info(f"Number of Atoms in fragment: {self.numatoms}")
        logger.info(f"Formula: {self.prettyformula}")
        logger.info(f"Label: {self.label}")
        logger.info(f"Charge: {self.charge} Mult: {self.mult}")
        logger.info("Do fragment.info() for more info on fragment")
        return "fragment"

    def info(self):
        """Log a summary of the fragment: formula, atom count, charge and multiplicity."""
        logger.info("Fragment object")
        logger.info("%s", self.__dict__)

    def update_attributes(self):
        """Recompute the derived attributes after the coordinates or elements change.

        Refreshes atom count, masses, nuclear charges, the molecular formula and the
        element list. Called by every method that replaces coordinates.
        """
        logger.info("Creating/Updating fragment attributes...")
        if len(self.coords) == 0:
            raise InputError("No coordinates in fragment. Something went wrong. Exiting.")
        if not isinstance(self.coords, np.ndarray):
            raise InputError("self.coords is not a numpy array. Something is wrong. Exiting.")
        self.nuccharge = nucchargelist(self.elems)
        self.nuc_charges = elemstonuccharges(self.elems)
        self.numatoms = len(self.coords)
        self.atomlist = list(range(self.numatoms))
        # Unnecessary alias ? Todo: Delete
        self.allatoms = self.atomlist
        self.mass = totmasslist(self.elems)
        self.list_of_masses = list_of_masses(self.elems)
        self.masses = self.list_of_masses
        # Elemental formula
        self.formula = elemlisttoformula(self.elems)
        # Pretty formula without 1 TODO
        self.prettyformula = self.formula
        # Update atomtypes, atomcharges and fragmenttype_labels also if needed
        if len(self.atomcharges) == 0:
            self.atomcharges = [0.0 for i in range(self.numatoms)]
        elif len(self.atomcharges) < self.numatoms:
            logger.warning("\natomcharges list shorter than number of atoms.")
            logger.info("Adding 0.0 entries for missing atoms.")
            self.atomcharges = self.atomcharges + [0.0 for i in range(self.numatoms - len(self.atomcharges))]

        if len(self.fragmenttype_labels) == 0:
            self.fragmenttype_labels = ["None" for i in range(self.numatoms)]
        elif len(self.fragmenttype_labels) < self.numatoms:
            logger.warning("\nfragmenttype_labels list shorter than number of atoms.")
            logger.info("Adding 0 entries for missing atoms.")
            self.fragmenttype_labels = self.fragmenttype_labels + [
                0 for i in range(self.numatoms - len(self.fragmenttype_labels))
            ]

        if len(self.atomtypes) == 0:
            self.atomtypes = ["None" for i in range(self.numatoms)]
        elif len(self.atomtypes) < self.numatoms:
            logger.warning("\natomtypes list shorter than number of atoms.")
            logger.info("Adding None entries for missing atoms.")
            self.atomtypes = self.atomtypes + ["None" for i in range(self.numatoms - len(self.atomtypes))]

        logger.info(f"Number of Atoms in fragment: {self.numatoms}\nFormula: {self.prettyformula}\nLabel: {self.label}")
        logger.info(f"Charge: {self.charge} Mult: {self.mult}")
        logger.info(sub_header_end())

    # Add coordinates from geometry string. Will replace.
    def add_coords_from_string(self, coordsstring, scale=None, tol=None, conncalc=False):
        """Append atoms parsed from a multi-line "El x y z" coordinate string.

        Args:
            coordsstring: coordinates, one "element x y z" line per atom.
            scale: covalent-radius scaling used for connectivity (defaults to the module setting).
            tol: covalent-radius tolerance used for connectivity.
            conncalc: recompute connectivity after adding the atoms.
        """
        logger.info("Getting coordinates from string: %s", coordsstring)
        if len(self.coords) > 0:
            logger.info("Fragment already contains coordinates")
            logger.info("Adding extra coordinates")
        coordslist = coordsstring.split("\n")
        tempcoords = []
        for line in coordslist:
            if len(line) > 5:
                self.elems.append(reformat_element(line.split()[0]))
                # Appending to numpy array
                clist = [float(line.split()[1]), float(line.split()[2]), float(line.split()[3])]
                tempcoords.append(clist)
        # Converting list of lists to numpy array
        self.coords = reformat_list_to_array(tempcoords)
        self.label = "".join(self.elems)
        # if conncalc is True:

    def create_coords_from_smiles(self, smiles):
        """Generate 3D coordinates from a SMILES string (requires OpenBabel).

        Args:
            smiles: SMILES representation of the molecule.
        """
        logger.info("Creating coordinates from SMILES string: %s", smiles)
        from openmmqmmm.openbabel import smiles_to_coords

        elems, coords = smiles_to_coords(smiles)
        self.elems = elems
        self.coords = reformat_list_to_array(coords)
        self.update_attributes()

    # Replace coordinates by providing elems and coords lists. Optional: recalculate connectivity
    def replace_coords(self, elems, coords, conn=False, scale=None, tol=None):
        """Replace the elements and coordinates with a new set.

        Used by the optimizers to install an updated geometry.

        Args:
            elems: element symbols, one per atom.
            coords: coordinates in Angstrom, one row per atom.
            conn: recompute connectivity afterwards.
            scale: covalent-radius scaling used for connectivity.
            tol: covalent-radius tolerance used for connectivity.
        """
        logger.info("Replacing coordinates in fragment.")

        self.elems = elems
        # Adding coords as list of lists. Conversion to numpy array
        self.coords = reformat_list_to_array(coords)
        self.update_attributes()
        if conn is True:
            self.calc_connectivity(scale=scale, tol=tol)

    def get_non_h_atomindices(self):
        """Return the indices of all atoms that are not hydrogen."""
        return [index for index, el in enumerate(self.elems) if el != "H"]

    def get_atomindices_for_element(self, element):
        """Return the indices of every atom of a given element.

        Args:
            element: element symbol, e.g. "Fe".
        """
        return [index for index, el in enumerate(self.elems) if el == element]

    def delete_atom(self, atomindex):
        """Remove one atom and refresh the derived attributes.

        Args:
            atomindex: index of the atom to delete.
        """
        self.coords = np.delete(self.coords, atomindex, axis=0)
        # Deleting from lists
        self.elems.pop(atomindex)
        self.atomcharges.pop(atomindex)
        self.atomtypes.pop(atomindex)
        self.fragmenttype_labels.pop(atomindex)

        # Updating other attributes
        self.update_attributes()

    def print_coords(self):
        """Log the coordinates of every atom in the fragment."""
        logger.info("Cartesian coordinates (Å):")
        for i, (el, c) in enumerate(zip(self.elems, self.coords, strict=False)):
            line = f" {i:<4} {el:4} {c[0]:>12.6f} {c[1]:>12.6f} {c[2]:>12.6f}"
            logger.info("%s", line)

    def print_coords_for_atoms(self, members, labels=None):
        """Log the coordinates of selected atoms.

        Args:
            members: atom indices to print.
            labels: optional per-atom labels shown alongside the coordinates.
        """
        print_coords_for_atoms(self.coords, self.elems, members, labels=labels)

    # Read Amber coordinate file? Needs to read both INPCRD and PRMTOP file. Bit messy
    def read_amberfile(self, inpcrdfile=None, prmtopfile=None, conncalc=False):
        """Read coordinates and topology from Amber inpcrd/prmtop files.

        Args:
            inpcrdfile: Amber coordinate file.
            prmtopfile: Amber topology file (supplies elements and masses).
            conncalc: compute connectivity after reading.
        """
        logger.info(
            f"Reading coordinates from Amber INPCRD file: '{inpcrdfile}' and PRMTOP file: '{prmtopfile}' into fragment."
        )
        try:
            elems, coords, _box_dims = read_ambercoordinates(prmtopfile=prmtopfile, inpcrdfile=inpcrdfile)
            # NOTE: boxdims not used. Could be set as fragment variable ?
        except FileNotFoundError:
            raise FileFormatError(f"File {prmtopfile} or {inpcrdfile} not found") from None
        self.coords = reformat_list_to_array(coords)
        self.elems = elems
        # if conncalc is True:

    # Read GROMACS coordinates file
    def read_grofile(self, filename, conncalc=False, scale=None, tol=None):
        """Read coordinates from a GROMACS .gro file.

        Args:
            filename: path to the .gro file.
            conncalc: compute connectivity after reading.
            scale: covalent-radius scaling used for connectivity.
            tol: covalent-radius tolerance used for connectivity.
        """
        logger.info(f"Reading coordinates from Gromacs GRO file '{filename}' into fragment")
        try:
            elems, coords, _boxdims = read_gromacsfile(filename)
            # NOTE: boxdims not used. Could be set as fragment variable ?
        except FileNotFoundError:
            raise FileFormatError(f"File '{filename}' not found") from None
        self.coords = coords
        self.elems = elems
        # if conncalc is True:

    # Read Chemshell fragment file (.c ending)
    def read_chemshellfile(self, filename, conncalc=False, scale=None, tol=None):
        """Read coordinates from a ChemShell fragment file (Bohr units).

        Args:
            filename: path to the ChemShell .c file.
            conncalc: compute connectivity after reading.
            scale: covalent-radius scaling used for connectivity.
            tol: covalent-radius tolerance used for connectivity.
        """
        logger.info(f"Reading coordinates from Chemshell file '{filename}' into fragment.")
        try:
            elems, coords = read_chemshellfragfile_xyz(filename)
        except FileNotFoundError:
            raise FileFormatError(f"File '{filename}' not found.") from None
        self.coords = coords
        self.elems = elems
        # if conncalc is True:
        #    # Read connectivity list

    def read_pdbfile_openmm(self, filename):
        """Read a PDB file using OpenMM's parser, keeping the full topology.

        Args:
            filename: path to the PDB file.
        """
        logger.info(f"read_pdbfile_openmm: Reading coordinates from PDB file '{filename}' into fragment.")
        try:
            import openmm.app
        except ImportError:
            raise FileFormatError(
                "Error: OpenMM library not found. the OpenMM library is required to read PDB files."
            ) from None
        pdb = openmm.app.PDBFile(filename)
        self.coords = np.array([[i.x * 10, i.y * 10, i.z * 10] for i in pdb.positions])
        self.elems = []
        logger.info("%s", pdb.topology)
        for atom in pdb.topology.atoms():
            if atom.element is None:
                # Virtual sites (the TIP4P M-site, for one) carry no element
                logger.warning("Could not fully parse element information from PDB-topology for atom: %s", atom)
                logger.info("This may be a virtual site. Adding 'M' as dummy element for this atom.")
                self.elems.append("M")
            else:
                self.elems.append(atom.element.symbol)

        # Topology
        self.pdb_topology = pdb.topology

    # Reading PDBx/mmCIF file using OpenMM
    def read_pdbxfile(self, filename):
        """Read a PDBx/mmCIF file using OpenMM's parser, keeping the full topology.

        Args:
            filename: path to the PDBx/mmCIF file.
        """
        logger.info(f"read_pdbxfile: Reading coordinates from PDBX file '{filename}' into fragment.")
        try:
            import openmm.app
        except ImportError:
            raise FileFormatError(
                "Error: OpenMM library not found. the OpenMM library is required to read PDB files."
            ) from None
        pdb = openmm.app.PDBxFile(filename)
        self.coords = np.array([[i.x * 10, i.y * 10, i.z * 10] for i in pdb.positions])
        self.elems = [atom.element.symbol for atom in pdb.topology.atoms()]

        # Topology
        self.pdb_topology = pdb.topology

    def read_xyzfile(self, filename, scale=None, tol=None, readchargemult=False, conncalc=True):
        """Read coordinates from an XYZ file.

        Args:
            filename: path to the XYZ file.
            scale: covalent-radius scaling used for connectivity.
            tol: covalent-radius tolerance used for connectivity.
            readchargemult: read charge and multiplicity from the title line.
            conncalc: compute connectivity after reading.
        """
        logger.info(f"Reading coordinates from XYZ file '{filename}' into fragment.")
        coords = []
        with open(filename) as f:
            for count, line in enumerate(f):
                if count == 0:
                    self.numatoms = int(line.split()[0])
                elif count == 1:
                    if readchargemult is True:
                        logger.info("Reading charge/mult from file header.")
                        try:
                            self.charge = int(line.split()[0])
                            self.mult = int(line.split()[1])
                        except ValueError:
                            raise FileFormatError(
                                "{}\nLine: {}".format(
                                    f"Error: XYZ-file {filename} does not have a valid charge/mult in 2nd-line of "
                                    f"header:",
                                    line,
                                )
                            ) from None
                elif count > 1 and len(line) > 3:
                    # Grabbing element and reformatting
                    if isint(line.split()[0]) is True:
                        # Grabbing element as atomnumber and reformatting
                        el = reformat_element(int(line.split()[0]), isatomnum=True)
                        self.elems.append(el)
                    else:
                        el = line.split()[0]
                        self.elems.append(reformat_element(el))
                    coords.append([float(line.split()[1]), float(line.split()[2]), float(line.split()[3])])
        # Convert to numpy
        self.coords = reformat_list_to_array(coords)
        if self.numatoms != len(self.coords):
            raise FileFormatError("Number of atoms in header not equal to number of coordinate-lines. Check XYZ file!")

    def set_energy(self, energy):
        """Store a total energy on the fragment.

        Args:
            energy: total energy in hartree.
        """
        self.energy = float(energy)

    def get_coordinate_center(self):
        """Return the geometric centre of the coordinates (unweighted by mass)."""
        center_x = np.mean(self.coords[:, 0])
        center_y = np.mean(self.coords[:, 1])
        center_z = np.mean(self.coords[:, 2])
        return [center_x, center_y, center_z]

    # Get coordinates for specific atoms (from list of atom indices)
    # NOTE: This also returns elements, bit silly
    def get_coords_for_atoms(self, atoms):
        """Return the coordinates and elements of a subset of atoms.

        Args:
            atoms: atom indices to extract.

        Returns:
            (subset_coords, subset_elems).
        """
        subcoords = np.take(self.coords, atoms, axis=0)
        subelems = [self.elems[i] for i in atoms]
        return subcoords, subelems

    # Calculate connectivity (list of lists) of coords
    def calc_connectivity(self, conndepth=99, scale=None, tol=None):
        """Compute the connectivity table and store it on the fragment.

        Args:
            conndepth: how many bonds outwards to follow when grouping atoms into molecules.
            scale: covalent-radius scaling used to decide whether two atoms are bonded.
            tol: covalent-radius tolerance added to the scaled radii.
        """
        logger.info("Calculating connectivity.")
        if len(self.coords) > 10000:
            logger.info("Atom number > 10K. Connectivity calculation could take a while")

        if scale is None:
            scale = CONNECTIVITY_SCALE
            tol = CONNECTIVITY_TOL
        logger.info(f"Using scale: {scale} and tol: {tol} ")

        # Setting scale and tol as part of object for future usage (e.g. QM/MM link atoms)
        self.scale = scale
        self.tol = tol

        # Calculate connectivity by looping over all atoms
        timestampA = time.time()
        fraglist = calc_conn_py(self.coords, self.elems, conndepth, scale, tol)
        log_time_since(timestampA, "calc connectivity py")
        self.connectivity = fraglist
        # Calculate number of atoms in connectivity list of lists
        conn_number_sum = 0
        for sublist in self.connectivity:
            conn_number_sum += len(sublist)
        if self.numatoms != conn_number_sum:
            raise InputError(
                f"Connectivity problem\nself.connectivity: {self.connectivity}\nconn_number_sum: "
                f"{conn_number_sum}\nself numatoms {self.numatoms}"
            )
        self.connected_atoms_number = conn_number_sum

    # Centroid
    def get_centroid(self):
        """Return the centroid (mean position) of all atoms."""
        return np.mean(self.coords, axis=0)

    # Write PDB-file
    def write_pdbfile(self, filename="Fragment"):
        """Write a PDB file using the fragment's own stored PDB information.

        Requires the fragment to have been created from a PDB file.

        Args:
            filename: output name, without the .pdb extension.
        """
        logger.info("Fragment.write_pdbfile method called")
        filename = filename.replace(".pdb", "")
        # Write PDB-file if information is available
        if self.pdb_atomnames is not None:
            logger.info("Found PDB residue/atom/segment information stored in fragment. Writing proper PDB file.")
        else:
            logger.warning(
                "Warning: No PDB residue/atom/segment information available (only available if Fragment was created "
                "from a PDB-file)."
            )
            logger.info("Will write PDB file with basic default residue/atom/segment names.")
        write_pdbfile(
            self,
            outputname=filename,
            atomnames=self.pdb_atomnames,
            chainlabels=self.pdb_chainlabels,
            resnames=self.pdb_resnames,
            residlabels=self.pdb_residlabels,
            segmentlabels=None,
            conect_lines=self.pdb_conect_lines,
        )
        return f"{filename}.pdb"

    # Create new topology from scratch if none is defined (defined automatically when reading PDB-files by OpenMM)
    def define_topology(self, scale=1.0, tol=0.1, resname="MOL"):
        """Build an OpenMM topology for the fragment from its connectivity.

        Args:
            scale: covalent-radius scaling used to detect bonds.
            tol: covalent-radius tolerance added to the scaled radii.
            resname: residue name given to every atom.
        """
        try:
            import openmm.app
        except ImportError:
            raise InputError("Error: OpenMM not found. Cannot define a topology") from None
        logger.info("Defining new basic single-chain, multi-residue topology")
        self.pdb_topology = openmm.app.Topology()
        chain = self.pdb_topology.addChain()

        # Create connectivity by default for new topology
        if self.connectivity is None or (isinstance(self.connectivity, list) and len(self.connectivity) == 0):
            self.calc_connectivity(scale=scale, tol=tol)

        connectivity_dict = get_connected_atoms_dict(self.coords, self.elems, scale, tol)
        # Looping over molecules defined by connectivity
        for mol in self.connectivity:
            logger.info("mol: %s", mol)
            residue = self.pdb_topology.addResidue(resname, chain)
            logger.info("residue: %s", residue)

            # Defaultdictionary to keep track of unique element-atomnames
            atomnames_dict = defaultdict(int)
            for at in mol:
                el = self.elems[at]
                atomnumber = openmm.app.Element.getBySymbol(el).atomic_number
                element = openmm.app.Element.getByAtomicNumber(atomnumber)
                # Define unique atomname
                atomnames_dict[el] += 1
                atomname = f"{el}{atomnames_dict[el]}"

                # Special handling for obvious water residues. Aids OpenMM recognition
                if atomname == "O1" and len(mol) == 3:
                    atomname = "O"
                logger.info("Adding atom: %s element: %s to residue: %s", atomname, element, residue)
                logger.info("at: %s el: %s", at, el)
                self.pdb_topology.addAtom(atomname, element, residue)

        logger.info("Adding connectivity to PDB topology")
        openmmqmmm.openmm.openmm_add_bonds_to_topology(self.pdb_topology, connectivity_dict)

        return self.pdb_topology

    # Write PDB-file via OpenMM
    def write_pdbfile_openmm(
        self, filename="Fragment", calc_connectivity=False, pdb_topology=None, skip_connectivity=False, resname="MOL"
    ):
        """Write a PDB file via OpenMM, building a topology if none is defined.

        Args:
            filename: output name, without the .pdb extension.
            calc_connectivity: recompute connectivity before writing.
            pdb_topology: existing OpenMM topology to use instead of the fragment's.
            skip_connectivity: write without CONECT records.
            resname: residue name used when a topology has to be created.
        """
        logger.info("write_pdbfile_openmm\n")
        try:
            import openmm.app
        except ImportError:
            raise InputError(
                "Error: OpenMM library not found. the OpenMM library is required to write PDB files."
            ) from None

        # Adding extension
        if ".pdb" not in filename:
            filename += ".pdb"

        if pdb_topology is not None:
            logger.info("Using input pdb_topology")
            self.pdb_topology = pdb_topology
        elif self.pdb_topology is None:
            logger.warning("Fragment has no PDB-file topology defined (required for PDB-file writing)")
            logger.info("Now defining new topology from scratch")
            if pdb_topology is None:
                self.define_topology(resname=resname)  # Creates self.pdb_topology
        else:
            logger.info("Using pdbtopology found in fragment")

        # Before writing PDB-file, request connectivity calculation so that we get correct CONECT lines for
        # non-biomolecules
        if calc_connectivity is True:
            logger.info("Connectivity calculation requested for Fragment")
            connectivity_dict = get_connected_atoms_dict(self.coords, self.elems, 1.0, 0.1)
            logger.info("Adding connectivity to PDB topology")
            openmmqmmm.openmm.openmm_add_bonds_to_topology(self.pdb_topology, connectivity_dict)

        # If no_connectivity is True, we skip adding connectivity to PDB-file
        if skip_connectivity is True:
            logger.info("skip_connectivity True: this will not write connectivity lines to PDB-file")
            logger.info("Deleting molecule bond information")
            # Setting list of bonds to empty list
            self.pdb_topology._bonds = []
        with open(filename, "w") as pdbhandle:
            openmm.app.PDBFile.writeFile(self.pdb_topology, self.coords, file=pdbhandle)
        logger.info(f"Wrote PDB-file: {filename}")
        return filename

    def write_xyzfile(
        self, xyzfilename="Fragment-xyzfile.xyz", writemode="w", write_chargemult=True, write_energy=True
    ):
        """Write the coordinates to an XYZ file.

        Args:
            xyzfilename: output file name.
            writemode: "w" to overwrite or "a" to append (for trajectories).
            write_chargemult: include charge and multiplicity in the title line.
            write_energy: include the stored energy in the title line.
        """
        with open(xyzfilename, writemode) as ofile:
            ofile.write(str(len(self.elems)) + "\n")
            # Title line
            # Write charge,mult and energy by default. Will be None if not available
            if write_chargemult is True and write_energy is True:
                ofile.write(f"{self.charge} {self.mult} {self.energy}\n")
            else:
                ofile.write("title\n")

            # Coordinates
            for el, c in zip(self.elems, self.coords, strict=False):
                line = f"{el:4} {c[0]:14.8f} {c[1]:14.8f} {c[2]:14.8f}"
                ofile.write(line + "\n")
        logger.info("Wrote XYZ file:  %s", xyzfilename)
        return xyzfilename

    def write_xyz_for_atoms(self, xyzfilename="Fragment-subset.xyz", atoms=None):
        """Write an XYZ file containing only the selected atoms.

        Args:
            xyzfilename: output file name.
            atoms: atom indices to write.
        """
        subset_elems = [self.elems[i] for i in atoms]
        subset_coords = np.take(self.coords, atoms, axis=0)
        with open(xyzfilename, "w") as ofile:
            ofile.write(str(len(subset_elems)) + "\n")
            ofile.write("title" + "\n")
            for el, c in zip(subset_elems, subset_coords, strict=False):
                line = f"{el:4} {c[0]:>12.6f} {c[1]:>12.6f} {c[2]:>12.6f}"
                ofile.write(line + "\n")

    # Print system-fragment information to file
    def print_system(self, filename="fragment.frag"):
        """Write the full fragment (coordinates, charge, mult, connectivity) to a .frag file.

        Args:
            filename: output file name.
        """
        logger.info("Printing fragment to disk:  %s", filename)
        logger.debug("len(self.atomlist):  %s", len(self.atomlist))
        logger.debug("len(self.elems):  %s", len(self.elems))
        logger.debug("len(self.coords):  %s", len(self.coords))
        logger.debug("len(self.atomcharges):  %s", len(self.atomcharges))
        logger.debug("len(self.fragmenttype_labels):  %s", len(self.fragmenttype_labels))
        logger.debug("len(self.atomtypes):  %s", len(self.atomtypes))

        if (
            len(self.atomlist)
            == len(self.elems)
            == len(self.coords)
            == len(self.atomcharges)
            == len(self.fragmenttype_labels)
            == len(self.atomtypes)
        ) is False:
            logger.error("Missing entries in list.")
            logger.info("Len atomlist: %s", len(self.atomlist))
            logger.info("Len elems: %s", len(self.elems))
            logger.info("Len coords: %s", len(self.coords))
            logger.info("Len atomcharges: %s", len(self.atomcharges))
            raise InternalError(
                f"Len atomtypes: {len(self.atomtypes)}\nLen fragmenttype_labels: "
                f"{len(self.fragmenttype_labels)}\nfragmenttype_labels: {self.fragmenttype_labels}\nThis should not "
                f"have happened. File a bugreport"
            )
        with open(filename, "w") as outfile:
            outfile.write("Fragment: \n")
            outfile.write(f"Num atoms: {self.numatoms}\n")
            outfile.write(f"Formula: {self.formula}\n")
            outfile.write(f"Energy: {self.energy}\n")
            if self.charge is not None:
                outfile.write(f"charge : {self.charge}\n")
            if self.mult is not None:
                outfile.write(f"mult : {self.mult}\n")
            outfile.write("\n")
            outfile.write(
                " Index    Atom         x                  y                  z               charge        "
                "fragment-type        atom-type\n"
            )
            outfile.write(
                "---------------------------------------------------------------------------------------------------------------------------------\n"
            )
            for at, el, coord, charge, label, atomtype in zip(
                self.atomlist,
                self.elems,
                self.coords,
                self.atomcharges,
                self.fragmenttype_labels,
                self.atomtypes,
                strict=False,
            ):
                label_str = str(label)
                line = (
                    f"{at:>6} {el:>6}  {coord[0]:17.11f}  {coord[1]:17.11f}  {coord[2]:17.11f}"
                    f"  {charge:14.8f} {label_str:12s} {atomtype:>21}\n"
                )
                outfile.write(line)
            outfile.write(
                "===========================================================================================================================================\n"
            )
            outfile.write(f"atomcharges: {self.atomcharges}\n")
            outfile.write(f"Sum of atomcharges: {sum(self.atomcharges)}\n")
            outfile.write(f"atomtypes: {self.atomtypes}\n")
            outfile.write(f"connectivity: {self.connectivity}\n")
            outfile.write(f"Centralmainfrag: {self.Centralmainfrag}\n")

    # Reading fragment from file. File created from Fragment.print_system
    def read_fragment_from_file(self, fragfile):
        """Load a fragment previously written by print_system.

        Args:
            fragfile: path to the .frag file.
        """
        logger.info("Reading fragment from file: %s", fragfile)
        coordgrab = False
        coords = []
        elems = []
        atomcharges = []
        atomtypes = []
        fragment_type_labels = []
        connectivity = []
        # Only used by molcrys:
        Centralmainfrag = []
        with open(fragfile) as file:
            for n, line in enumerate(file):
                if n == 0 and "Fragment:" not in line:
                    raise FileFormatError("This is not a valid fragment file. Exiting.")
                if "Num atoms:" in line:
                    int(line.split()[-1])
                if "charge :" in line:
                    self.charge = int(line.split()[-1])
                if "mult :" in line:
                    self.mult = int(line.split()[-1])
                if coordgrab is True:
                    # If end of coords section
                    if "===============" in line:
                        coordgrab = False
                        continue
                    elems.append(line.split()[1])
                    coords.append([float(line.split()[2]), float(line.split()[3]), float(line.split()[4])])
                    atomcharges.append(float(line.split()[5]))
                    # Reading and converting to integer.
                    ftypelabel = "None" if line.split()[6] == "None" else int(line.split()[6])
                    fragment_type_labels.append(ftypelabel)
                    atomtypes.append(line.split()[7])

                if "--------------------------" in line:
                    coordgrab = True
                if "Centralmainfrag" in line and "[]" not in line:
                    payload = line.removeprefix("Centralmainfrag:")
                    for junk in ("\n", " ", "[", "]"):
                        payload = payload.replace(junk, "")
                    Centralmainfrag = [int(i) for i in payload.split(",")]
                # Incredibly ugly but oh well
                if "connectivity:" in line:
                    payload = line.removeprefix("connectivity:")
                    payload = payload.replace(" ", "")
                    for x in payload.split("]"):
                        if len(x) < 1:
                            break
                        y = x.strip(",[")
                        y = y.strip("[")
                        y = y.strip("]")
                        try:
                            connlist = [int(i) for i in y.split(",")]
                        except ValueError:
                            connlist = []
                        connectivity.append(connlist)
        self.elems = elems
        # Converting to numpy array
        self.coords = np.array(coords)
        self.atomcharges = atomcharges
        self.atomtypes = atomtypes
        self.fragmenttype_labels = fragment_type_labels
        self.update_attributes()
        self.connectivity = connectivity
        self.Centralmainfrag = Centralmainfrag


def reformat_list_to_array(data):
    """Return coordinates as an (N, 3) array, accepting an array or a list of lists.

    Anything else is rejected rather than returned as None: this feeds Fragment.coords,
    and a None there surfaces much later as an unrelated error.
    """
    # If np array already
    if isinstance(data, np.ndarray):
        return data
    # Reformat to np array
    if isinstance(data, list):
        # Checking if input is a list of lists or not
        if any(isinstance(el, list) for el in data) is False:
            raise InputError("Error (reformat_list_to_array): input should be a list of lists, not just a list")
        return np.array(data)
    raise InputError(
        "Error (reformat_list_to_array): coordinates must be a list of lists or a numpy array, "
        f"got {type(data).__name__}"
    )


# Function to reformat element string to be correct('cu' or 'CU' become 'Cu')
# Can also convert atomic-number (isatomnum flag)
def reformat_element(elem, isatomnum=False):
    if isatomnum is True:
        try:
            el_correct = openmmqmmm.elements.element_dict_atnum[elem].symbol
        except KeyError:
            raise InputError(
                f"Element-string {elem} is not a valid element. Fix the element information in the coordinate file."
            ) from None
    else:
        try:
            el_correct = openmmqmmm.elements.element_dict_atname[elem.lower()].symbol
        except KeyError:
            raise InputError(
                f"Element-string {elem} is not a valid element. Fix the element information in the coordinate file."
            ) from None
    return el_correct


_DEFAULT_RADIUS = 1.50  # fallback for elements missing from eldict_covrad
_CONNECTIVITY_TOLERANCE = 0.40  # Angstrom added to sum of covalent radii


def _build_connectivity(coords, elems, atom_indices=None):
    coords = np.asarray(coords)
    n = len(elems)

    # Same radii as the other connectivity paths (threshold_conn, get_connected_atoms_np):
    # eldict_covrad carries the Na/K and M-site overrides that keep ions and TIP4P dummy
    # sites from bonding to their neighbours. This used to be a second, unmodified copy of
    # the Alvarez table, so the two paths disagreed about exactly those atoms.
    radii = np.array([eldict_covrad.get(e.capitalize(), _DEFAULT_RADIUS) for e in elems])

    # Keep full-length connectivity list so downstream code
    # can continue using global atom indices
    conn = [set() for _ in range(n)]

    # Default behaviour: full-system connectivity
    atom_indices = range(n) if atom_indices is None else list(atom_indices)

    nsel = len(atom_indices)

    for a in range(nsel):
        i = atom_indices[a]

        for b in range(a + 1, nsel):
            j = atom_indices[b]

            dist = np.linalg.norm(coords[i] - coords[j])
            threshold = radii[i] + radii[j] + _CONNECTIVITY_TOLERANCE

            # Ignore very short distances
            if 0.4 < dist < threshold:
                conn[i].add(j)
                conn[j].add(i)

    return conn


# NEW function to print internal coordinate table for active atoms based on connectivity.


def _print_internal_coordinate_table(fragment, actatoms=None):
    """Print a tabulated view of internal coordinates for the active atoms.

    The table is built from the fragment's connectivity.
    """

    def _measure_bond(coords, i, j):
        """Bond length in Angstrom between atoms i and j."""
        return float(np.linalg.norm(coords[i] - coords[j]))

    def _measure_angle(coords, i, j, k):
        """Angle i-j-k in degrees (j is the vertex)."""
        v1 = coords[i] - coords[j]
        v2 = coords[k] - coords[j]
        cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        return float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))

    def _measure_dihedral(coords, i, j, k, l):  # noqa: E741 - dihedral atoms i-j-k-l
        """Dihedral angle i-j-k-l in degrees (range -180 to 180)."""
        b1 = coords[j] - coords[i]
        b2 = coords[k] - coords[j]
        b3 = coords[l] - coords[k]
        n1 = np.cross(b1, b2)
        n2 = np.cross(b2, b3)
        m1 = np.cross(n1, b2 / np.linalg.norm(b2))
        return float(np.degrees(np.arctan2(np.dot(m1, n2), np.dot(n1, n2))))

    if actatoms is None:
        actatoms = fragment.allatoms

    coords = fragment.coords
    elems = fragment.elems
    conn = _build_connectivity(coords, elems)

    # Header
    logger.info("")
    logger.info("%s", "=" * 30)
    logger.info("Internal Coordinates")
    logger.info("%s", "=" * 30)
    logger.info(f"{'Type':<10} {'Atoms':<20} {'Elements':<15} {'Value':>10}")
    logger.info("%s", "-" * 60)

    # We use sets to avoid printing the same geometric feature twice
    # (e.g., bond 0-1 and 1-0)
    seen_bonds = set()
    seen_angles = set()
    seen_dihedrals = set()

    for i in actatoms:
        # --- Bonds (i-j) ---
        for j in conn[i]:
            bond_key = tuple(sorted((i, j)))
            if bond_key not in seen_bonds:
                val = _measure_bond(coords, i, j)
                label = f"{elems[i]}-{elems[j]}"
                logger.info(f"{'Bond':<10} {bond_key!s:<20} {label:<15} {val:>10.4f} Å")
                seen_bonds.add(bond_key)

            # --- Angles (i-j-k) ---
            # i is the vertex (j-i-k)
            neighbors = list(conn[i])
            for idx_a in range(len(neighbors)):
                for idx_b in range(idx_a + 1, len(neighbors)):
                    n_a, n_b = neighbors[idx_a], neighbors[idx_b]
                    angle_key = (*sorted((n_a, n_b)), i)  # vertex last for keying
                    if angle_key not in seen_angles:
                        val = _measure_angle(coords, n_a, i, n_b)
                        label = f"{elems[n_a]}-{elems[i]}-{elems[n_b]}"
                        logger.info(f"{'Angle':<10} {f'({n_a},{i},{n_b})':<20} {label:<15} {val:>10.2f}°")
                        seen_angles.add(angle_key)

        # --- Dihedrals (i-j-k-l) ---
        # Logic: Find a bond (i-j), then find neighbors of i and j
        for j in conn[i]:
            for h in conn[i]:
                if h == j:
                    continue
                for k in conn[j]:
                    if k in (i, h):
                        continue
                    # Path is h-i-j-k
                    di_key = (h, i, j, k)
                    rev_key = (k, j, i, h)
                    if di_key not in seen_dihedrals and rev_key not in seen_dihedrals:
                        val = _measure_dihedral(coords, h, i, j, k)
                        label = f"{elems[h]}-{elems[i]}-{elems[j]}-{elems[k]}"
                        logger.info(f"{'Dihedral':<10} {di_key!s:<20} {label:<15} {val:>10.2f}°")
                        seen_dihedrals.add(di_key)

    logger.info("%s", "-" * 60)


# OLD FUNCTION.
def print_internal_coordinate_table(fragment, actatoms=None) -> None:
    """Log a table of bonds, angles and dihedrals for a fragment.

    Args:
        fragment: Fragment to analyze. Connectivity is recalculated as needed.
        actatoms: optional list of atom indices to restrict the table to.
    """
    timeA = time.time()
    logger.info("\nPrinting internal coordinate table")
    if actatoms is not None:
        logger.info("Actatoms: %s", actatoms)

    # If no actatoms
    if actatoms is None:
        actatoms = []
        chosen_coords = fragment.coords
        chosen_elems = fragment.elems

    # NOTE: Changing so that we calculate connectivity always regardless of availability.
    # If no connectivity in fragment then recalculate it for actatoms only
    # if len(fragment.connectivity) == 0:
    logger.info("Connectivity needs to be calculated")

    if len(actatoms) > 0:
        chosen_coords = np.take(fragment.coords, actatoms, axis=0)
        chosen_elems = [fragment.elems[i] for i in actatoms]
    else:
        chosen_coords = fragment.coords
        chosen_elems = fragment.elems

    conndepth = 99
    scale = CONNECTIVITY_SCALE
    tol = CONNECTIVITY_TOL

    connectivity = calc_conn_py(chosen_coords, chosen_elems, conndepth, scale, tol)
    logger.info("Connectivity calculation complete.")

    # Looping over connected fragments
    bondpairsdict = {}

    for conn_fragment in connectivity:
        # Looping over atom indices in fragment
        for atom in conn_fragment:
            connatoms = get_connected_atoms(chosen_coords, chosen_elems, CONNECTIVITY_SCALE, CONNECTIVITY_TOL, atom)
            for conn_i in connatoms:
                dist = distance(chosen_coords[atom], chosen_coords[conn_i])
                bondpairsdict[frozenset((atom, conn_i))] = dist

    logger.info(small_header("Internal coordinates"))

    # Using frozenset: https://stackoverflow.com/questions/46633065/multiples-keys-dictionary-where-key-order-doesnt-matter
    logger.info(small_header("Bond lengths (Å):"))
    for key, val in bondpairsdict.items():
        listkey = list(key)
        elA = chosen_elems[listkey[0]]
        elB = chosen_elems[listkey[1]]
        # Only print bond lengths if both atoms in actatoms list
        if not actatoms:
            logger.info(f"Bond: {listkey[0]:8}{elA:4} - {listkey[1]:4}{elB:4} {val:>6.3f}")
        else:
            # converting to full-system indices
            fullsystem_keyA = actatoms[listkey[0]]
            fullsystem_keyB = actatoms[listkey[1]]
            if fullsystem_keyA in actatoms and fullsystem_keyB in actatoms:
                logger.info(f"Bond: {fullsystem_keyA:8}{elA:4} - {fullsystem_keyB:4}{elB:4} {val:>6.3f}")
    logger.info("%s", "=" * 50)
    log_time_since(timeA, "print internal coordinate table")


# From lists of coords,elems and atom indices, print coords with elem
def print_coords_for_atoms(coords, elems, members, labels=None):
    if labels is not None and len(labels) != len(members):
        raise InputError("Problem. Length of Labels note equal to length of members list")
    label = ""
    for i, m in enumerate(members):
        if labels is not None:
            label = labels[i]
        logger.info(f"{label:>4} {elems[m]:>4} {coords[m][0]:>12.8f}  {coords[m][1]:>12.8f}  {coords[m][2]:>12.8f}")


# From lists of coords,elems and atom indices, write XYZ file coords with elem


def write_xyz_for_atoms(coords, elems, members, name):
    subset_elems = [elems[i] for i in members]
    subset_coords = np.take(coords, members, axis=0)
    with open(name + ".xyz", "w") as ofile:
        ofile.write(str(len(subset_elems)) + "\n")
        ofile.write("title" + "\n")
        for el, c in zip(subset_elems, subset_coords, strict=False):
            line = f"{el:4} {c[0]:>12.6f} {c[1]:>12.6f} {c[2]:>12.6f}"
            ofile.write(line + "\n")


# From lists of coords,elems and atom indices, print coords with elems
# If list of atom indices provided, print as leftmost column
# If list of labels provided, print as rightmost column
# If list of labels2 provided, print as rightmost column
def print_coords_all(coords, elems, indices=None, labels=None, labels2=None):
    if indices is None:
        if labels is None:
            for i in range(len(elems)):
                logger.info(f"{elems[i]:>4} {coords[i][0]:>12.8f}  {coords[i][1]:>12.8f}  {coords[i][2]:>12.8f}")
        elif labels2 is None:
            for i in range(len(elems)):
                logger.info(
                    f"{elems[i]:>4} {coords[i][0]:>12.8f}  {coords[i][1]:>12.8f}  {coords[i][2]:>12.8f} {labels[i]:>6}"
                )
        else:
            for i in range(len(elems)):
                logger.info(
                    f"{elems[i]:>4} {coords[i][0]:>12.8f}  {coords[i][1]:>12.8f}  {coords[i][2]:>12.8f} "
                    f"{labels[i]:>6} {labels2[i]:>6}"
                )
    elif labels is None:
        for i in range(len(elems)):
            logger.info(
                f"{indices[i]:>1} {elems[i]:>4} {coords[i][0]:>12.8f}  {coords[i][1]:>12.8f}  {coords[i][2]:>12.8f}"
            )
    elif labels2 is None:
        for i in range(len(elems)):
            logger.info(
                f"{indices[i]:>1} {elems[i]:>4} {coords[i][0]:>12.8f}  {coords[i][1]:>12.8f}  "
                f"{coords[i][2]:>12.8f} {labels[i]:>6}"
            )
    else:
        for i in range(len(elems)):
            logger.info(
                f"{indices[i]:>1} {elems[i]:>4} {coords[i][0]:>12.8f}  {coords[i][1]:>12.8f}  "
                f"{coords[i][2]:>12.8f} {labels[i]:>6} {labels2[i]:>6}"
            )


# From lists of coords,elems and atom indices, print coords with elems
# If list of atom indices provided, print as leftmost column
# If list of labels provided, print as rightmost column
# If list of labels2 provided, print as rightmost column
def write_coords_all(coords, elems, indices=None, labels=None, labels2=None, file="file", description="description"):
    with open(file, "w") as f:
        _write_coords_lines(f, coords, elems, indices, labels, labels2, description)


def _write_coords_lines(f, coords, elems, indices, labels, labels2, description):
    f.write(f"#{description}\n")
    if indices is None:
        if labels is None:
            f.writelines(
                f"{elems[i]:>4} {coords[i][0]:>12.8f}  {coords[i][1]:>12.8f}  {coords[i][2]:>12.8f}\n"
                for i in range(len(elems))
            )

        elif labels2 is None:
            f.writelines(
                f"{elems[i]:>4} {coords[i][0]:>12.8f}  {coords[i][1]:>12.8f}  {coords[i][2]:>12.8f} {labels[i]:>6}\n"
                for i in range(len(elems))
            )
        else:
            f.writelines(
                f"{elems[i]:>4} {coords[i][0]:>12.8f}  {coords[i][1]:>12.8f}  {coords[i][2]:>12.8f} {labels[i]:>6} "
                f"{labels2[i]:>6}\n"
                for i in range(len(elems))
            )
    elif labels is None:
        f.writelines(
            f"{indices[i]:>1} {elems[i]:>4} {coords[i][0]:>12.8f}  {coords[i][1]:>12.8f}  {coords[i][2]:>12.8f}\n"
            for i in range(len(elems))
        )
    elif labels2 is None:
        f.writelines(
            f"{indices[i]:>1} {elems[i]:>4} {coords[i][0]:>12.8f}  {coords[i][1]:>12.8f}  "
            f"{coords[i][2]:>12.8f} {labels[i]:>6}\n"
            for i in range(len(elems))
        )
    else:
        f.writelines(
            f"{indices[i]:>1} {elems[i]:>4} {coords[i][0]:>12.8f}  {coords[i][1]:>12.8f}  "
            f"{coords[i][2]:>12.8f} {labels[i]:>6} {labels2[i]:>6}\n"
            for i in range(len(elems))
        )


##############################################################
# Functions to get distance, angle, coordinates of fragment
##############################################################


def distance(A, B):
    return sqrt((A[0] - B[0]) ** 2 + (A[1] - B[1]) ** 2 + (A[2] - B[2]) ** 2)  # fastest


def angle(A, B, C):
    AB = A - B
    CB = C - B
    dot_product = np.dot(AB, CB)
    # Calculate the magnitudes of the vectors
    magnitude1 = np.linalg.norm(AB)
    magnitude2 = np.linalg.norm(CB)
    # Calculate the angle in radians
    angle_rad = np.arccos(dot_product / (magnitude1 * magnitude2))
    # Convert angle to degrees
    return np.degrees(angle_rad)


def dihedral(A, B, C, D):
    # Calculate the vectors between adjacent atoms
    v1 = B - A
    v2 = C - B
    v3 = D - C

    # Calculate the cross products
    n1 = np.cross(v1, v2)
    n2 = np.cross(v2, v3)

    # Calculate the dot product
    dot = np.dot(n1, n2)
    # Handle signs correctly
    if dot < 0:
        dihedral_angle = -1 * (np.arccos(dot / (np.linalg.norm(n1) * np.linalg.norm(n2))))
    else:
        dihedral_angle = np.arccos(dot / (np.linalg.norm(n1) * np.linalg.norm(n2)))

    # Convert from radians to degrees
    return dihedral_angle * 180 / np.pi


# User-functions
# atoms is a list of atom indices,
def distance_between_atoms(fragment=None, atoms=None) -> float:
    """Return the distance between two atoms of a fragment.

    Args:
        fragment: Fragment holding the coordinates.
        atoms: list of 2 atom indices (0-based).

    Returns:
        Distance in Angstrom.
    """
    return distance(fragment.coords[atoms[0]], fragment.coords[atoms[1]])


def angle_between_atoms(fragment=None, atoms=None) -> float:
    """Return the A-B-C angle spanned by three atoms of a fragment.

    Args:
        fragment: Fragment holding the coordinates.
        atoms: list of 3 atom indices (0-based); the middle one is the vertex.

    Returns:
        Angle in degrees.
    """
    return angle(fragment.coords[atoms[0]], fragment.coords[atoms[1]], fragment.coords[atoms[2]])


def dihedral_between_atoms(fragment=None, atoms=None) -> float:
    """Return the A-B-C-D dihedral angle spanned by four atoms of a fragment.

    Args:
        fragment: Fragment holding the coordinates.
        atoms: list of 4 atom indices (0-based), in bonding order.

    Returns:
        Signed dihedral angle in degrees.
    """
    return dihedral(
        fragment.coords[atoms[0]], fragment.coords[atoms[1]], fragment.coords[atoms[2]], fragment.coords[atoms[3]]
    )


# TODO: clean up
def get_centroid(coords):
    sum_x = 0
    sum_y = 0
    sum_z = 0
    for c in coords:
        sum_x += c[0]
        sum_y += c[1]
        sum_z += c[2]
    return [sum_x / len(coords), sum_y / len(coords), sum_z / len(coords)]


# Change origin to centroid. Either use centroid of full system (default) or alternatively subset or (something else
# even)
def change_origin_to_centroid(fullcoords, subsetcoords=None, subsetatoms=None):
    if subsetcoords is not None:
        logger.info("Calculating centroid for the specified subset coordinates")
        centroid = get_centroid(subsetcoords)
    elif subsetatoms is not None:
        logger.info("Calculating centroid for the coordintes of specified subatoms: %s", subsetatoms)
        # Will grab subsetcoords
        subcoords = np.take(fullcoords, subsetatoms, axis=0)
        centroid = get_centroid(subcoords)
    else:
        logger.info("Calculating centroid for full set of coordinates")
        centroid = get_centroid(fullcoords)

    newcoords = fullcoords - centroid
    logger.info("Returning full coordinates with new origin at centroid")
    return newcoords


# Determine threshold for whether atoms are connected or not based on covalent radii for pair of atoms
# Uses global scale and tol parameters that may be changed at input
def threshold_conn(elA, elB, scale, tol):
    return scale * (eldict_covrad[elA] + eldict_covrad[elB]) + tol


# Connectivity function (called by Fragment object)
def calc_conn_py(coords, elems, conndepth, scale, tol):
    found_atoms = []
    fraglist = []
    for atom in range(len(elems)):
        if atom not in found_atoms:
            members = get_molecule_members_loop_np2(coords, elems, conndepth, scale, tol, atomindex=atom)
            if members not in fraglist:
                fraglist.append(members)
                found_atoms += members
    return fraglist


# Get connected atoms to chosen atom index based on threshold
# Uses slow for-loop structure with distance-function call
# Don't use unless system is small
def get_connected_atoms(coords, elems, scale, tol, atomindex):
    connatoms = []
    coords_ref = coords[atomindex]
    elem_ref = elems[atomindex]
    for i, c in enumerate(coords):
        if distance(coords_ref, c) < threshold_conn(elems[i], elem_ref, scale, tol) and i != atomindex:
            connatoms.append(i)
    return connatoms


# Euclidean distance functions:
# https://semantive.com/pl/blog/high-performance-computation-in-python-numpy/
def einsum_mat(mat_v, mat_u):
    mat_z = mat_v - mat_u
    return np.sqrt(np.einsum("ij,ij->i", mat_z, mat_z))


# Get connected atoms to chosen atom index based on threshold
# np version for calculating the euclidean distance
# https://semantive.com/pl/blog/high-performance-computation-in-python-numpy/
def get_connected_atoms_np(coords, elems, scale, tol, atomindex):
    # Creating np array of the coords to compare
    compcoords = np.tile(coords[atomindex], (len(coords), 1))
    # All distances in one go
    distances = einsum_mat(coords, compcoords)
    # Getting all thresholds as list via list comprehension.
    el_covrad_ref = eldict_covrad[elems[atomindex]]
    # Cheaper way of getting thresholds list than calling threshold_conn
    # List comprehension of dict lookup and convert to numpy. Should be as fast as can be done
    # for i in range(len(thresholds)):
    # TODO: Slowest part but hard to make faster
    thresholds = np.array([eldict_covrad[elems[i]] for i in range(len(elems))])
    # Numpy addition and multiplication done on whole array
    thresholds = thresholds + el_covrad_ref
    thresholds = thresholds * scale
    thresholds = thresholds + tol
    # Old slow way
    # Getting difference of distances and thresholds
    diff = distances - thresholds
    # Getting connatoms by finding indices of diff with negative values (i.e. where distance is smaller than threshold)
    return np.where(diff < 0)[0].tolist()


# Get a dictionary of atoms (values) connected to each atom (key)
def get_connected_atoms_dict(coords, elems, scale, tol):
    conndict = {}
    for c in range(len(coords)):
        conn = get_connected_atoms_np(coords, elems, scale, tol, c)
        conn.remove(c)
        conndict[c] = conn
    return conndict


# Numpy clever loop test.
# Version 2 never goes through same atom


def get_molecule_members_loop_np2(coords, elems, loopnumber, scale, tol, atomindex=None, membs=None):
    if membs is None:
        membs = []
        membs.append(atomindex)
        membs = get_connected_atoms_np(coords, elems, scale, tol, atomindex)

    # If membs is just an integer turn into list
    if isinstance(membs, int):
        membs = [membs]
    finalmembs = membs

    for _i in range(loopnumber):
        # Get list of lists of connatoms for each member
        newmembers = [get_connected_atoms_np(coords, elems, scale, tol, k) for k in membs]
        # Get a unique flat list
        trimmed_flat = np.unique([item for sublist in newmembers for item in sublist]).tolist()

        # Check if new atoms not previously found
        membs = listdiff(trimmed_flat, finalmembs)
        # Exit loop if nothing new found
        if len(membs) == 0:
            return finalmembs
        finalmembs += membs
        finalmembs = np.unique(finalmembs).tolist()
    return finalmembs


# Takes list of elements and gives formula
def elemlisttoformula(elems):
    """Build a molecular formula string from a list of element symbols.

    Uses Hill notation — carbon first, then hydrogen, then the remaining elements
    alphabetically — so the same system always gives the same string. Iterating a set
    instead made the formula differ between processes (Python randomises string
    hashing), and the formula is part of the calculation labels in singlepoint.py.

    Args:
        elems: element symbols, one per atom.

    Returns:
        Formula string, e.g. "H2O1" for water and "C2H6O1" for ethanol.
    """
    # Counting once per unique element rather than per atom: elems can be very long
    counts = Counter(elems)
    ordered = []
    if "C" in counts:
        ordered.append("C")
        if "H" in counts:
            ordered.append("H")
    ordered += sorted(element for element in counts if element not in ordered)
    return "".join(f"{element}{counts[element]}" for element in ordered)


# From molecular formula (string, e.g. "FeCl4") to list of atoms
def molformulatolist(formulastring):
    el = ""
    diff = ""
    els = []
    atomunits = []
    numels = []
    # Read string by character backwards
    for _count, char in enumerate(formulastring[::-1]):
        if isint(char):
            el = char + el
        if char.islower():
            el = char + el
            diff = char + diff
        if char.isupper():
            el = char + el
            diff = char + diff
            atomunits.append(el)
            els.append(diff)
            el = ""
            diff = ""
    for atm, element in zip(atomunits, els, strict=False):
        if atm > element:
            number = atm[len(element) :]
            numels.append(int(number))
        else:
            number = 1
            numels.append(int(number))
    atoms = [element for element, count in zip(els, numels, strict=False) for _ in range(count)]
    # Final reverse
    els.reverse()
    numels.reverse()
    atoms.reverse()
    return atoms


# Read XYZ file
def read_xyzfile(filename) -> tuple[list[str], np.ndarray]:
    """Read elements and coordinates from an XYZ file.

    The element column may hold either element symbols or atomic numbers.

    Args:
        filename: path to the XYZ file.

    Returns:
        Tuple (elems, coords): list of element symbols and a list of
        [x, y, z] coordinate triples in Angstrom.

    Raises:
        FileFormatError: if the atom count in the header does not match the
            number of coordinate lines, or the elements and coordinates
            disagree in length.
    """
    # Will accept atom-numbers as well as symbols
    logger.info(f"Reading coordinates from XYZ file '{filename}'.")
    coords = []
    elems = []
    with open(filename) as f:
        for count, line in enumerate(f):
            if count == 0:
                numatoms = int(line.split()[0])
            if count > 1 and len(line.strip()) > 0:
                if isint(line.split()[0]) is True:
                    # Grabbing element as atomnumber and reformatting
                    el = reformat_element(int(line.split()[0]), isatomnum=True)
                    elems.append(el)
                else:
                    # Grabbing element as symbol and reformatting just in case
                    el = reformat_element(line.split()[0])
                    elems.append(el)
                coords.append([float(line.split()[1]), float(line.split()[2]), float(line.split()[3])])
    if len(coords) != numatoms:
        raise FileFormatError(
            f"Error: Number of coordinates in XYZ-file: {filename} does not match header line. Exiting."
        )
    if len(coords) != len(elems):
        raise FileFormatError(
            f"Number of coordinates does not match elements. Something wrong with XYZ-file?:  {filename}"
        )
    return elems, coords


# Read all XYZ-files from directory
# Return fragment list
def read_xyzfiles(xyzdir, readchargemult=False, label_from_filename=True) -> list:
    """Create a Fragment for every XYZ file in a directory.

    Files are processed in natural (human) sort order.

    Args:
        xyzdir: directory to scan for *.xyz files.
        readchargemult: read charge and multiplicity from each XYZ title line.
        label_from_filename: unused; the filename is always used as the label.

    Returns:
        List of Fragment objects, one per file.
    """
    logger.info("read_xyzfiles function")
    logger.info("Note: will read XYZ-files in directory using natural sorting")
    import glob

    filenames = []
    fragments = []
    for file in natural_sort(glob.glob(xyzdir + "/*.xyz")):
        filename = os.path.basename(file)
        filenames.append(filename)
        logger.info("\n\nXYZ-file: %s", filename)
        # Creating new fragment, reading charge/mult and using filename as fragment label
        mol = openmmqmmm.Fragment(xyzfile=file, readchargemult=readchargemult, label=filename)
        fragments.append(mol)
    return fragments


# Write XYZfile provided list of elements and list of list of coords and filename
# Fast version. Note: list comprehension is bottleneck, unclear how to make this faster though
def write_xyzfile(elems, coords, name, writemode="w", title="title") -> None:
    """Write elements and coordinates to an XYZ file.

    Args:
        elems: list of element symbols.
        coords: sequence of [x, y, z] coordinate triples in Angstrom.
        name: output basename; ".xyz" is appended.
        writemode: file mode, "w" to overwrite or "a" to append a frame.
        title: text for the XYZ title (second) line.
    """
    # Adding headerlines to list
    header = [f"{len(elems)}\n", f"{title}\n"]
    atomlines = [f"{el:4} {c[0]:16.12f} {c[1]:16.12f} {c[2]:16.12f}\n" for el, c in zip(elems, coords, strict=False)]
    with open(name + ".xyz", writemode) as ofile:
        ofile.writelines(header)
        ofile.writelines(atomlines)
    logger.info("Wrote XYZ file:  %s", name + ".xyz")


# Function that reads XYZ-file with multiple files, splits and return list of coordinates
# Created for splitting crest_conformers.xyz but may also be used for MD traj.
# Also grabs last word in title line. Typically an energy (has to be converted to float outside)
def split_multimolxyzfile(file, writexyz=False, skipindex=1, return_fragments=False) -> list | tuple[list, list, list]:
    """Split a multi-molecule XYZ file (trajectory, conformer set) into its frames.

    Args:
        file: path to the multi-molecule XYZ file.
        writexyz: also write each frame to its own molecule<N>.xyz file.
        skipindex: keep only every Nth frame.
        return_fragments: return Fragment objects instead of raw arrays.

    Returns:
        A list of Fragment objects if return_fragments is True, otherwise the
        tuple (all_elems, all_coords, all_titles) with one entry per frame.
    """
    all_coords = []
    all_elems = []
    all_titles = []
    molcounter = 0
    coordgrab = False
    titlegrab = False
    coords = []
    elems = []
    fragments = []
    with open(file) as f:
        for index, line in enumerate(f):
            if index == 0:
                numatoms = line.split()[0]
            # Grab coordinates
            if coordgrab is True:
                if len(line.split()) > 1:
                    elems.append(reformat_element(line.split()[0]))
                    coords_x = float(line.split()[1])
                    coords_y = float(line.split()[2])
                    coords_z = float(line.split()[3])
                    coords.append([coords_x, coords_y, coords_z])
                if len(coords) == int(numatoms):
                    all_coords.append(coords)
                    all_elems.append(elems)
                    if writexyz is True:
                        # Alternative option: write each conformer/molecule to disk as XYZfile
                        write_xyzfile(elems, coords, "molecule" + str(molcounter))
                    frag = Fragment(coords=coords, elems=elems)
                    fragments.append(frag)
                    coords = []
                    elems = []
            # Grab title
            if titlegrab is True:
                if len(line.split()) > 0:
                    all_titles.append(line.split())
                else:
                    all_titles.append("NA")
                titlegrab = False
                coordgrab = True
            # Grabbing number of atoms from string
            if len(line.split()) > 0 and line.split()[0] == str(numatoms):
                if molcounter % skipindex:
                    molcounter += 1
                    titlegrab = False
                    coordgrab = False
                else:
                    molcounter += 1
                    titlegrab = True
                    coordgrab = False
    logger.info(f"Found {molcounter} geometries in file: {file}")

    if return_fragments is True:
        return fragments
    return all_elems, all_coords, all_titles


# Read Tcl-Chemshell fragment file and grab elems and coords. Coordinates converted from Bohr to Angstrom
def read_chemshellfragfile_xyz(fragfile):
    # removing extension from fragfile name if present and then adding back.
    pathtofragfile = fragfile.split(".")[0] + ".c"
    coords = []
    elems = []
    # TODO: Change elems and coords to numpy array instead
    grabcoords = False
    with open(pathtofragfile) as ffile:
        for line in ffile:
            if "block = connectivity" in line:
                grabcoords = False
            if grabcoords is True:
                coords.append([float(i) * openmmqmmm.constants.bohr2ang for i in line.split()[1:]])
                el = reformat_element(line.split()[0])
                elems.append(el)
            if "block = coordinates records " in line:
                grabcoords = True
        coords = reformat_list_to_array(coords)
    return elems, coords


def conv_atomtypes_elems(atomtype):
    """Convert a forcefield atomtype string to an element symbol.

    Falls back to treating the atomtype as an element symbol (with case
    normalization) when it is not in the atomtype dictionary.

    Args:
        atomtype: forcefield atomtype or element string, e.g. "HA" or "FE".

    Returns:
        Element symbol, e.g. "H" or "Fe".
    """
    try:
        return openmmqmmm.elements.atomtypes_dict[atomtype]
    except KeyError:
        # Assume correct element but could be wrongly formatted (e.g. FE instead of Fe) so reformatting
        try:
            return reformat_element(atomtype)
        except InputError:
            raise InputError(
                (
                    "{}\nYou might have to modify the atomtype/element information in "
                    "coordinate file you're reading in."
                ).format(f"Atomtype: '{atomtype}' not recognized either as valid atomtype or element. Exiting.")
            ) from None


# READ PDBfile
def read_pdbfile(filename, use_atomnames_as_elements=False):
    residuelist = []
    # If elemcolumn found
    elemcol = []
    # Not atomtype but atomname
    residname = []

    coords = []
    try:
        with open(filename) as f:
            for line in f:
                # if 'ATOM ' in line or 'HETATM' in line:
                if line.startswith(("ATOM", "HETATM")):
                    atom_name = line[12:16].replace(" ", "")
                    residname.append(line[17:20].replace(" ", ""))
                    residuelist.append(line[22:26].replace(" ", ""))
                    coords_x = float(line[30:38].replace(" ", ""))
                    coords_y = float(line[38:46].replace(" ", ""))
                    coords_z = float(line[46:54].replace(" ", ""))
                    coords.append([coords_x, coords_y, coords_z])
                    elem = line[76:78].replace(" ", "").replace("\n", "")
                    # Option to use atomnamecolumn for element information instead of element-column
                    if use_atomnames_as_elements is True:
                        elem_name = openmmqmmm.elements.atomtypes_dict[atom_name]
                        elemcol.append(elem_name)
                    elif len(elem) != 0:
                        if len(elem) == 2:
                            # Making sure second elem letter is lowercase
                            elemcol.append(reformat_element(elem))
                        else:
                            elemcol.append(reformat_element(elem))
                    else:
                        logger.info("While reading line:")
                        raise FileFormatError(
                            f"{line}\nNo element found in element-column of PDB-file\nEither fix element-column "
                            f"(columns 77-78) or try to use to read element-information from atomname-column:\n "
                            f"Fragment(pdbfile='X', use_atomnames_as_elements=True)"
                        )
                # if 'HETATM' in line:
    except FileNotFoundError:
        raise FileFormatError(f"File '{filename}' does not exist!") from None
    # Create numpy array
    coords_np = reformat_list_to_array(coords)

    if len(elemcol) != len(coords):
        raise FileFormatError(
            f"len coords {len(coords)}\nlen elemcol {len(elemcol)}\ndid not find same number of elements as "
            f"coordinates\nNeed to define elements in some other way"
        )
    elems = elemcol
    return elems, coords_np


def read_pdbfile_info(filename, use_atomnames_as_elements=False):
    atomnames = []
    chainlabels = []
    residnames = []
    residlabels = []
    conect_lines = []
    try:
        with open(filename) as f:
            for line in f:
                if line.startswith(("ATOM", "HETATM")):
                    atomnames.append(line[12:16].replace(" ", ""))
                    residnames.append(line[17:20].replace(" ", ""))
                    chainlabels.append(line[21:22].replace(" ", ""))
                    # Resid grab
                    # Note: Resids are integer up to 9999 but after that many programs (VMD, OpenMM) switch to a hex
                    # notation
                    # Here grabbing resid as string instead of integer in general
                    residlabel_temp = line[22:26].replace(" ", "")
                    if residlabel_temp == "A000":
                        logger.warning(
                            "Warning: read_pdbfile_info encountered a hexadecimal notation (A000) for resid (likely "
                            "due to resids > 9999). Hopefully things will be fine"
                        )
                        logger.info(f"PDB-file: {filename}. Line: {line}")
                    residlabel = str(residlabel_temp)
                    residlabels.append(residlabel)
                if line.startswith("CONECT"):
                    conect_lines.append(line)
    except FileNotFoundError:
        raise FileFormatError(f"File '{filename}' does not exist!") from None

    return atomnames, residnames, residlabels, chainlabels, conect_lines


# Read GROMACS Gro coordinate file and box info
# Read AMBERCRD file and coords and box info
# Not part of Fragment class because we don't have element information here
def read_gromacsfile(grofile) -> tuple[list[str], np.ndarray, list]:
    """Read a GROMACS .gro coordinate file.

    Args:
        grofile: path to the .gro file.

    Returns:
        Tuple (elems, coords, box_dims): element symbols guessed from the atom
        names, an (natoms, 3) numpy array of coordinates converted from nm to
        Angstrom, and [a, b, c, 90.0, 90.0, 90.0] cell parameters in Angstrom
        (the cell is assumed orthorhombic) or None if the file had no box line.

    Raises:
        FileFormatError: if the parsed elements and coordinates disagree in length.
    """
    elems = []
    coords = []
    # TODO: Change coords to numpy array instead
    numatoms = "unset"
    box_dims = None
    with open(grofile) as cfile:
        for i, line in enumerate(cfile):
            if i == 0:
                pass
            elif i == 1:
                numatoms = int(line.split()[0])
                logger.info("Numatoms: %s", numatoms)
            elif i == numatoms + 2:
                # Last line: box dimensions
                box_dims = [10 * float(i) for i in line.split()]
                # Assuming cubic and adding 90,90,90
                box_dims.append(90.0)
                box_dims.append(90.0)
                box_dims.append(90.0)
                logger.info("Box dimensions read:  %s", box_dims)
            else:
                linelist = line.split()
                # Grabbing atomtype
                atomtype = linelist[1]
                atomtype = "".join(item for item in atomtype if not item.isdigit())
                atomtype = atomtype.replace("'", "")
                # Converting atomtype to element based on function above
                elem = conv_atomtypes_elems(atomtype)
                elems.append(elem)

                # If larer than 7 then GRO file contains both coords and velocities
                if len(linelist) > 7:
                    coords_x = float(linelist[-6])
                    coords_y = float(linelist[-5])
                    coords_z = float(linelist[-4])
                # If smaller then only coords
                else:
                    coords_x = float(linelist[-3])
                    coords_y = float(linelist[-2])
                    coords_z = float(linelist[-1])
                # Converting from nm to Ang
                coords.append([10 * coords_x, 10 * coords_y, 10 * coords_z])
    npcoords = reformat_list_to_array(coords)
    if len(npcoords) != len(elems):
        raise FileFormatError(f"Num coords not equal to num elems. Parsing of Gromacsfile: {grofile} failed. BUG!")
    return elems, npcoords, box_dims


# Read AMBERCRD file and coords and box info
# Not part of Fragment class because we don't have element information here
def read_ambercoordinates(prmtopfile=None, inpcrdfile=None) -> tuple[list[str], np.ndarray, list]:
    """Read an Amber inpcrd/rst coordinate file, taking elements from the prmtop.

    Args:
        prmtopfile: Amber topology file, used for the element list.
        inpcrdfile: Amber coordinate file (inpcrd or restart).

    Returns:
        Tuple (elems, coords, box_dims): element symbols, an (natoms, 3) numpy
        array of coordinates in Angstrom, and the cell parameters from the
        final line (empty list if the file had no box line).
    """
    elems = []
    coords = []
    # TODO: Change coords to numpy array instead
    numatoms = "unset"
    box_dims = []
    with open(inpcrdfile) as cfile:
        for i, line in enumerate(cfile):
            if i == 0:
                pass
            elif i == 1:
                numatoms = int(line.split()[0])
                logger.info("Numatoms:  %s", numatoms)
                numcoordlines = math.ceil(numatoms / 2)
            elif i == numcoordlines + 2:
                # Last line: box dimensions
                box_dims = [float(i) for i in line.split()]
                logger.info("Box dimensions read:  %s", box_dims)
            else:
                linelist = line.split()
                coordvalues = []
                # Checking if values combined: e,g, -16.3842161-100.0326085
                # Then split and add
                for c in linelist:
                    if c.count(".") > 1:
                        d = c.replace("-", " -").split()
                        coordvalues.append(float(d[0]))
                        coordvalues.append(float(d[1]))
                    else:
                        coordvalues.append(float(c))
                coords.append([coordvalues[0], coordvalues[1], coordvalues[2]])
                if len(coordvalues) == 6:
                    coords.append([coordvalues[3], coordvalues[4], coordvalues[5]])

    # Grab atom numbers and convert to elements
    grab_atomnumber = False
    with open(prmtopfile) as pfile:
        for i, line in enumerate(pfile):
            if grab_atomnumber is True and "FORMAT" not in line:
                if "%" in line:
                    grab_atomnumber = False
                else:
                    elems += [reformat_element(int(i), isatomnum=True) for i in line.split()]
            if "%FLAG ATOMIC_NUMBER" in line:
                grab_atomnumber = True
    if len(coords) != len(elems):
        raise FileFormatError(
            f"Num coords ({len(coords)}) not equal to num elems ({len(elems)}). Parsing of Amber files: {prmtopfile} "
            f"and {inpcrdfile} failed. BUG!"
        )
    return elems, coords, box_dims


# Write PDBfile proper
# Example,manual: write_pdbfile(frag, outputname="name", atomnames=openmmobject.atomnames,
# resnames=openmmobject.resnames, residlabels=openmmobject.resids,segmentlabels=openmmobject.segmentnames)
# Example, simple: write_pdbfile(frag, outputname="name", openmmobject=objname)
# Example, minimal: write_pdbfile(frag)
# TODO: Add option to write new hybrid-36 standard PDB file instead of current hexadecimal nonstandard fix
def write_pdbfile(
    fragment,
    outputname="fragment",
    openmmobject=None,
    atomnames=None,
    resnames=None,
    residlabels=None,
    chainlabels=None,
    segmentlabels=None,
    dummyname="DUM",
    charges_column=None,
    conect_lines=None,
) -> str:
    """Write a fragment to a PDB file.

    Per-atom PDB columns are taken from an OpenMMTheory topology when
    openmmobject is given, otherwise from the explicit *labels arguments, and
    otherwise filled with generic defaults.

    Args:
        fragment: Fragment holding elements and coordinates.
        outputname: output basename; ".pdb" is appended.
        openmmobject: OpenMMTheory whose topology supplies atom/residue/chain labels.
        atomnames: per-atom names, overriding the topology.
        resnames: per-atom residue names, overriding the topology.
        residlabels: per-atom residue numbers, overriding the topology.
        chainlabels: per-atom chain identifiers, overriding the topology.
        segmentlabels: per-atom segment identifiers, overriding the topology.
        dummyname: residue name used for dummy atoms.
        charges_column: optional per-atom values written into the occupancy column.
        conect_lines: optional pre-built CONECT records to append.

    Returns:
        The name of the written file.
    """
    logger.info("Writing PDB-file...")
    # Using fragment
    elems = fragment.elems
    coords = fragment.coords

    # Can grab everything from OpenMMobject if provided
    # NOTE: These lists are only defined for CHARMM files currently. Not Amber or GROMACS
    if openmmobject is not None:
        atomnames = openmmobject.atomnames
        resnames = openmmobject.resnames
        residlabels = openmmobject.resids
        segmentlabels = openmmobject.segmentnames

    # What to choose if keyword arguments not given
    if atomnames is None or len(atomnames) == 0:
        logger.warning("Using elements as atomnames")
        # Elements instead. Means VMD will display atoms properly at least
        atomnames = fragment.elems
    if resnames is None or len(resnames) == 0:
        resnames = fragment.numatoms * [dummyname]
    if chainlabels is None or len(chainlabels) == 0:
        chainlabels = fragment.numatoms * [""]
    if residlabels is None or len(residlabels) == 0:
        residlabels = fragment.numatoms * [1]
    # Note: choosing to make segment ID 3-letter-string (and then space)
    if segmentlabels is None or len(segmentlabels) == 0:
        logger.warning("No segment labels found")
        segmentlabels = fragment.numatoms * ["   "]

    if len(atomnames) > 99999:
        logger.info("System larger than 99999 atoms. Will use hexadecimal notation for atom indices 100K and larger. ")

    if (len(atomnames) == len(coords) == len(resnames) == len(residlabels) == len(segmentlabels)) is False:
        logger.error("Something went wrong in write_pdbfile. Exiting. File a bug report.")
        logger.error("Problem with lists...")
        logger.info("len: atomnames %s", len(atomnames))
        logger.info("len: coords %s", len(coords))
        raise InternalError(
            f"len: resnames {len(resnames)}\nlen: residlabels {len(residlabels)}\nlen: segmentlabels "
            f"{len(segmentlabels)}\nlen elems: {len(elems)}"
        )

    with open(outputname + ".pdb", "w") as pfile:
        for count, (atomname, c, resname, chainlabel, resid, _seg, el) in enumerate(
            zip(atomnames, coords, resnames, chainlabels, residlabels, segmentlabels, elems, strict=False)
        ):
            atomindex = count + 1
            # Convert to hexadecimal if >= 100K.
            # Note: unsupported standard but VMD will read it
            atomindexstring = f"{count + 1:x}" if atomindex >= 100000 else str(atomindex)

            # Using only first 3 letters of RESname
            resname_short = resname[0:3]

            # Using last 4 letters of atomnmae
            atomnamestring = atomname[-4:]

            if not any(char.isdigit() for char in atomnamestring):
                atomnamestring = atomnamestring + str(count + 1)

            # Using string format from: cupnet.net/pdb-format/

            # NOTE: Changed resid from integer to string so that we can support the hex notation for resids when resids
            # go above 9999
            resid_str = str(resid)

            # Optional charges column (used by CP2K)
            if charges_column is not None:
                charge = charges_column[count]
                #    seg[0:3], el, charge)
                line = (
                    "{:6s}{:5s} {:^4s}{:1s}{:3s} {:1s}{:4s}{:1s}   "
                    "{:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}          {:>2s}{:2s}"
                ).format(
                    "ATOM",
                    atomindexstring,
                    atomnamestring,
                    "",
                    resname_short,
                    chainlabel,
                    resid_str,
                    "",
                    c[0],
                    c[1],
                    c[2],
                    1.0,
                    0.00,
                    el,
                    charge,
                )
            # Regular
            else:
                #    seg[0:3], el)
                line = (
                    "{:6s}{:5s} {:^4s}{:1s}{:3s} {:1s}{:4s}{:1s}   "
                    "{:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}          {:>2s}{:2s}"
                ).format(
                    "ATOM",
                    atomindexstring,
                    atomnamestring,
                    "",
                    resname_short,
                    chainlabel,
                    resid_str,
                    "",
                    c[0],
                    c[1],
                    c[2],
                    1.0,
                    0.00,
                    el,
                    "",
                )
            pfile.write(line + "\n")
        # Write CONECT lines if provided
        if conect_lines is not None:
            pfile.writelines(conect_lines)
    logger.info("Wrote PDB file:  %s", outputname + ".pdb")
    return outputname + ".pdb"


# Calculate total nuclear charge from list of elements
def nucchargelist(ellist):
    totnuccharge = 0
    warning_issued = False
    for e in ellist:
        try:
            atcharge = elematomnumbers[e.lower()]
        except KeyError:
            atcharge = 0.0
            if warning_issued is False:
                logger.warning(f"Unknown element: '{e}' found in element-list")
                logger.info("Could be dummy atom. Using nuccharge of 0.0")
                warning_issued = True
        totnuccharge += atcharge
    return totnuccharge


# get list of nuclear charges from list of elements
# Used by Psi4 and CM5calc and xTBlibrary
# aka atomic numbers, aka atom numbers
def elemstonuccharges(ellist):
    nuccharges = []
    for e in ellist:
        atcharge = elematomnumbers[e.lower()]
        nuccharges.append(atcharge)
    return nuccharges


# Calculate molecular mass from list of atoms
def totmasslist(ellist):
    return sum(list_of_masses(ellist))


# Calculate list of masses from list of elements
def list_of_masses(ellist):
    masses = []
    warning_issued = False
    for e in ellist:
        try:
            atcharge = int(elematomnumbers[e.lower()])
            if atcharge == 0:
                logger.warning(
                    f"Warning: element '{e}' has atomic number 0. This is likely a dummy atom. Using mass of 0.0"
                )
                atmass = 0.0
            elif atcharge > len(atommasses):
                # atommasses stops at Lr (Z=103); indexing past it would wrap round to a
                # random lighter element instead of failing.
                raise InputError(
                    f"No atomic mass available for element '{e}' (Z={atcharge}). "
                    f"The mass table covers Z=1-{len(atommasses)}."
                )
            else:
                atmass = atommasses[atcharge - 1]
        except KeyError:
            atmass = 0.0
            if warning_issued is False:
                logger.warning(f"Unknown element: '{e}' found in element-list")
                logger.info("Could be dummy atom. Using mass of 0.0")
                warning_issued = True
        masses.append(atmass)
    return masses


##########################################################################################################
# Flexible align functions that take 2 fragments or files and aligns one of them on top of the other
# Support different order, i.e. we reorder. We can either keep reorder or not
# We also support subset of atoms to align on.
##########################################################################################################


# For XYZ-files
def flexible_align_xyz(
    xyzfile_a, xyzfile_b, rotate_only=False, translate_only=False, reordering=False, reorder_method="brute", subset=None
) -> None:
    """Align the molecule in one XYZ file onto the molecule in another.

    Writes the aligned structure to <xyzfile_a stem>_aligned.xyz.

    Args:
        xyzfile_a: XYZ file to move.
        xyzfile_b: XYZ file to align onto (stays fixed).
        rotate_only: apply only the rotation, not the translation.
        translate_only: apply only the translation, not the rotation.
        reordering: reorder the atoms of A to match B before aligning.
        reorder_method: reordering algorithm, e.g. "brute" or "hungarian".
        subset: atom indices to align on; a list of two lists gives separate
            indices for A and B.
    """
    logger.info(f"Will align molecule in file {xyzfile_a} onto molecule in file {xyzfile_b}")
    fragment_a = Fragment(xyzfile=xyzfile_a)
    fragment_b = Fragment(xyzfile=xyzfile_b)

    newfragA = flexible_align(
        fragment_a,
        fragment_b,
        rotate_only=rotate_only,
        translate_only=translate_only,
        reordering=reordering,
        reorder_method=reorder_method,
        subset=subset,
    )

    # Write XYZ-file for newfragA
    newfragA.write_xyzfile(f"{xyzfile_a.replace('.xyz', '')}_aligned.xyz")


# For PDB-files
def flexible_align_pdb(
    pdbfileA, pdbfileB, rotate_only=False, translate_only=False, reordering=False, reorder_method="brute", subset=None
) -> None:
    """Align the molecule in one PDB file onto the molecule in another.

    Writes the aligned structure to <pdbfileA stem>_aligned.pdb, preserving the
    PDB metadata read from pdbfileA.

    Args:
        pdbfileA: PDB file to move.
        pdbfileB: PDB file to align onto (stays fixed).
        rotate_only: apply only the rotation, not the translation.
        translate_only: apply only the translation, not the rotation.
        reordering: reorder the atoms of A to match B before aligning.
        reorder_method: reordering algorithm, e.g. "brute" or "hungarian".
        subset: atom indices to align on; a list of two lists gives separate
            indices for A and B.
    """
    logger.info(f"Will align molecule in file {pdbfileA} onto molecule in file {pdbfileB}")
    fragment_a = Fragment(pdbfile=pdbfileA)
    fragment_b = Fragment(pdbfile=pdbfileB)

    # Call flexible align, get aligned coords as new fragA
    newfragA = flexible_align(
        fragment_a,
        fragment_b,
        rotate_only=rotate_only,
        translate_only=translate_only,
        reordering=reordering,
        reorder_method=reorder_method,
        subset=subset,
    )

    # Write PDBfile. PDB-info will have been read and stored
    fragment_a.coords = newfragA.coords  # Replacing coords in original fragmentA
    fragment_a.write_pdbfile_openmm(filename=f"{pdbfileA.replace('.pdb', '')}_aligned")  # Now write out


# For fragments
def flexible_align(
    fragment_a,
    fragment_b,
    rotate_only=False,
    translate_only=False,
    reordering=False,
    reorder_method="brute",
    subset=None,
) -> "Fragment":
    """Align one fragment onto another (Kabsch superposition, optional reordering).

    Args:
        fragment_a: Fragment to move.
        fragment_b: Fragment to align onto (stays fixed).
        rotate_only: apply only the rotation, not the translation.
        translate_only: apply only the translation, not the rotation.
        reordering: reorder the atoms of A to match B before aligning.
        reorder_method: reordering algorithm, e.g. "brute" or "hungarian".
        subset: atom indices to align on; a list of two lists gives separate
            indices for A and B.

    Returns:
        A new Fragment holding the aligned coordinates of fragment_a.
    """
    logger.info("flexible_align function")
    import geometric

    # Do chosen subset
    if subset is not None:
        logger.info("Subset option chosen")
        if any(isinstance(el, list) for el in subset) is True:
            logger.info("Subset is a list of lists")
            logger.info("Subset for A: %s", subset[0])
            logger.info("Subset for B: %s", subset[1])
            if len(subset[0]) != len(subset[1]):
                raise InputError("Length of subsets not equal. This is not allowed. Exiting.")
            logger.info("Will align using each list of indices for each fragment")
            subsetA_coords, subsetA_elems = fragment_a.get_coords_for_atoms(subset[0])
            subsetB_coords, subsetB_elems = fragment_b.get_coords_for_atoms(subset[1])

        else:
            logger.info("Subset is a list of indices")
            logger.info(
                "Will align using the same indices in both fragments (will only work if both fragments have the same "
                "atom order)"
            )
            subsetA_coords, subsetA_elems = fragment_a.get_coords_for_atoms(subset)
            subsetB_coords, subsetB_elems = fragment_b.get_coords_for_atoms(subset)

        logger.info("subsetA_elems: %s", subsetA_elems)
        logger.info("subsetA_coords: %s", subsetA_coords)

        logger.info("subsetB_elems: %s", subsetB_elems)
        logger.info("subsetB_coords: %s", subsetB_coords)

    else:
        subsetA_coords = fragment_a.coords
        subsetB_coords = fragment_b.coords

    # TODO Possible reordering
    if reordering is True:
        logger.info("Reordering atoms in fragmentB for better alignment (may not always work)")
        logger.warning("This requires the rmsd package to be installed: pip install rmsd")
        from rmsd import (
            reorder_brute,
            reorder_distance,
            reorder_hungarian,
            reorder_inertia_hungarian,
            reorder_similarity,
        )

        logger.info(f"Reorder method: {reorder_method}")
        if reorder_method == "brute":
            logger.warning("The brute force method can be very slow for large systems but is very accurate")
            logger.info("If too slow then try next (in order): inertia_hungarian, hungarian and distance")
        # Note: brute works well, hungarian fails e.g. for benzamidine example, distance works for benzamidine
        reorder_methods_dict = {
            "brute": reorder_brute,
            "hungarian": reorder_hungarian,
            "inertia_hungarian": reorder_inertia_hungarian,
            "similarity": reorder_similarity,
            "distance": reorder_distance,
        }
        logger.info(
            "Note: All reorder-method options (from rmsd pakcage): brute, hungarian, inertia_hungarian, similarity, "
            "distance"
        )
        order = reorder(
            reorder_methods_dict[reorder_method],
            np.array(subsetA_coords),
            np.array(subsetB_coords),
            np.array(subsetA_elems),
            np.array(subsetB_elems),
        )
        logger.info("Order: %s", order)
        subsetB_coords = subsetB_coords[order]
    else:
        logger.info("No reordering of atoms in fragmentA")

    # Use geometric function to get translation and rotation matrices for the subsets
    trans, rot = geometric.molecule.get_rotate_translate(subsetA_coords, subsetB_coords)

    # Translate only (all atoms in A)
    if translate_only is True:
        logger.info("Doing translation only")
        Anew = fragment_a.coords + trans
    # Rotate only (all atoms in A)
    elif rotate_only is True:
        logger.info("Doing rotation only")
        Anew = np.dot(fragment_a.coords, rot)
    else:
        # Apply trans+rot to all atoms in fragmentA
        Anew = np.dot(fragment_a.coords, rot) + trans

    # Create new frag
    newfrag = Fragment(elems=fragment_a.elems, coords=Anew)
    logger.info("New aligned structure")
    newfrag.print_coords()

    return newfrag


# Recommended RMSD-calc wrapper function for fragments
# Allows subset match (same set of indices or 2 sets of indices for each fragment)
# Also simpler option: heavyatomsonly=True (ignores H-atoms)
# NOTE: no reordering
def calculate_rmsd(fragment_a, fragment_b, subset=None, heavyatomsonly=False, write_aligned_structure=False) -> float:
    """Return the RMSD between two fragments after optimal superposition.

    Args:
        fragment_a: first Fragment.
        fragment_b: second Fragment.
        subset: atom indices to compare; a list of two lists gives separate
            indices for A and B.
        heavyatomsonly: exclude hydrogen atoms from the comparison.
        write_aligned_structure: also write the aligned structure to disk.

    Returns:
        RMSD in Angstrom.

    Raises:
        InputError: if the two subsets differ in length.
    """
    logger.info("calculate_RMSD function")

    # Do chosen subset
    if subset is not None:
        logger.info("Subset option chosen")
        if any(isinstance(el, list) for el in subset) is True:
            logger.info("Subset is a list of lists")
            logger.info("Subset for A: %s", subset[0])
            logger.info("Subset for B: %s", subset[1])
            if len(subset[0]) != len(subset[1]):
                raise InputError("Length of subsets not equal. This is not allowed. Exiting.")
            logger.info("Will align using each list of indices for each fragment")
            subsetA_coords, subsetA_elems = fragment_a.get_coords_for_atoms(subset[0])
            subsetB_coords, subsetB_elems = fragment_b.get_coords_for_atoms(subset[1])

        else:
            logger.info("Subset is a list of indices")
            logger.info(
                "Will align using the same indices in both fragments (will only work if both fragments have the same "
                "atom order)"
            )
            subsetA_coords, subsetA_elems = fragment_a.get_coords_for_atoms(subset)
            subsetB_coords, subsetB_elems = fragment_b.get_coords_for_atoms(subset)

        logger.debug("subsetA_elems: %s", subsetA_elems)
        logger.debug("subsetA_coords: %s", subsetA_coords)
        logger.debug("subsetB_elems: %s", subsetB_elems)
        logger.debug("subsetB_coords: %s", subsetB_coords)
    elif heavyatomsonly is True:
        subsetA_coords = fragment_a.coords[fragment_a.get_non_h_atomindices()]
        subsetB_coords = fragment_b.coords[fragment_b.get_non_h_atomindices()]

    else:
        subsetA_coords = fragment_a.coords
        subsetB_coords = fragment_b.coords

    # Use geometric function to get translation and rotation matrices for the subsets
    import geometric

    trans, rot = geometric.molecule.get_rotate_translate(subsetA_coords, subsetB_coords)
    Anew = np.dot(subsetA_coords, rot) + trans

    # RMSD
    rmsdval = float(np.sqrt(((Anew - subsetB_coords) ** 2).sum() / len(Anew)))

    logger.info("RMSD: %s", rmsdval)

    if write_aligned_structure:
        logger.info("write_aligned_structure active")
        newfrag = Fragment(elems=fragment_a.elems, coords=Anew)
        newfrag.write_xyzfile("structA_aligned.xyz")

    return rmsdval


#####################################
# RMSD and align related functions
#####################################


def centroid(X):
    """Compute the centroid of a vectorset.

    The centroid is the mean position of all the points in all of the coordinate
    directions: C = sum(X)/len(X). See https://en.wikipedia.org/wiki/Centroid.

    Args:
        X: (N,D) matrix, where N is points and D is dimension.

    Returns:
        The centroid, as a (D,) array.
    """
    return X.mean(axis=0)


def rmsd(V, W):
    """Calculate Root-mean-square deviation from two sets of vectors V and W."""
    D = len(V[0])
    N = len(V)
    rmsd = 0.0
    for v, w in zip(V, W, strict=False):
        rmsd += sum([(v[i] - w[i]) ** 2.0 for i in range(D)])
    return np.sqrt(rmsd / N)


# Get partial list by deleting elements not present in provided list of indices.
def get_partial_list(allatoms, partialatoms, full_list):
    newlist = copy.copy(full_list)  # Otherwise object may be updated
    otheratoms = listdiff(allatoms, partialatoms)
    otheratoms.reverse()
    for at in otheratoms:
        del newlist[at]
    return newlist


# Hungarian reorder algorithm
# From RMSD


def reorder(reorder_method, p_coord, q_coord, p_atoms, q_atoms):
    p_cent = centroid(p_coord)
    q_cent = centroid(q_coord)
    p_coord -= p_cent
    q_coord -= q_cent

    # Convert from element string to atomic number
    p_atoms = np.array([elematomnumbers[el.lower()] for el in p_atoms])
    q_atoms = np.array([elematomnumbers[el.lower()] for el in q_atoms])

    q_review = reorder_method(p_atoms, q_atoms, p_coord, q_coord)
    return [q_review.tolist()][0]


# QM-region expand function. Finds whole fragments.
def expand_qm_region(fragment=None, initial_atoms=None, radius=None) -> list[int]:
    # If needed (connectivity ==0) -> list[int]:
    """Expand a QM region outward to include whole molecules within a distance cutoff."""
    scale = CONNECTIVITY_SCALE
    tol = CONNECTIVITY_TOL
    if fragment is None or initial_atoms is None or radius is None:
        raise InputError("Provide fragment, initial_atoms and radius keyword arguments to QMregionfragexpand!")
    subsetcoords = np.take(fragment.coords, initial_atoms, axis=0)
    if len(fragment.connectivity) == 0:
        logger.info("No connectivity found. Using slow way of finding nearby fragments...")
    atomlist = []

    for c in subsetcoords:
        for index, allc in enumerate(fragment.coords):
            if index >= len(subsetcoords):
                dist = distance(c, allc)
                if dist < radius:
                    # Get molecule members atoms for atom index.
                    # Using stored connectivity because takes forever otherwise
                    # If no connectivity
                    if len(fragment.connectivity) == 0:
                        wholemol = get_molecule_members_loop_np2(
                            fragment.coords, fragment.elems, 99, scale, tol, atomindex=index
                        )

                    # If stored connectivity
                    else:
                        for q in fragment.connectivity:
                            if index in q:
                                wholemol = q
                                break

                    atomlist = atomlist + wholemol
    return np.unique(atomlist).tolist()


# Function to do QM-region expansion based on QM/MM pointcharge gradient
def expand_qm_pc_region(theory=None, fragment=None, thresh=5e-4) -> list[int]:
    """Expand a QM region based on the QM/MM pointcharge-gradient magnitude."""
    if theory is None and fragment is None:
        raise InputError("QMPC_fragexpand requires fragment and theory")
    if not isinstance(theory, openmmqmmm.QMMMTheory):
        raise InputError("Theory is not a QMMMTheory")

    # QM/MM run
    openmmqmmm.single_point(theory=theory, fragment=fragment, grad=True)

    # Selection scheme based on pointcharge gradient
    pcgrad = theory.PCgradient
    large_force_indices = np.unique(np.argwhere(abs(pcgrad) > thresh)[:, 0])
    # Convert pcgrad indices to system indices
    proper_largeforce_indices = large_force_indices + len(theory.qmatoms)
    # Get whole molecules
    fragment.calc_connectivity()  # get connectivity

    # New expansion
    new_expansion = theory.qmatoms

    for i in proper_largeforce_indices:
        mol_index = search_list_of_lists_for_index(i, fragment.connectivity)
        molmembers = fragment.connectivity[mol_index]
        new_expansion = new_expansion + molmembers
    new_expansion = np.unique(new_expansion)

    # Print to output and disk
    logger.info("New QM-region expansion based on pointcharge gradient selection")
    fragment.print_coords_for_atoms(new_expansion, labels=new_expansion)
    logger.info("Writing coordinates to file: QMPC_selection.xyz")
    fragment.write_xyz_for_atoms(xyzfilename="QMPC_selection.xyz", atoms=new_expansion)

    return new_expansion


# Function to determine the QM-MM boundary
# Note: This function was dominating QMMMTheory creation (e.g. 9.67 s / 12.41 s => 78 % for 300K system)
# Now sped up via get_connected_atoms_np. Silly
def get_boundary_atoms(qmatoms, coords, elems, scale, tol, excludeboundaryatomlist=None, unusualboundary=False):
    timeA = time.time()
    logger.info("Determining QM-MM/HL-LL boundary")
    logger.info("Parameters determing connectivity:")
    logger.info("Scaling factor: %s", scale)
    logger.info("Tolerance: %s", tol)
    if excludeboundaryatomlist is None:
        excludeboundaryatomlist = []

    logger.info("QM atoms: %s", qmatoms)
    logger.info(
        "QM atoms to be excluded from boundary creation (excludeboundaryatomlist):  %s", excludeboundaryatomlist
    )
    # For each QM atom, do a get_conn_atoms, for those atoms, check if atoms are in qmatoms,
    # if not, then we have found an MM-boundary atom

    # TODO: Note, there can can be problems here if either scale, tol is non-ideal value (should be set in inputfile)
    # TODO: Or if eldict_covrad needs to be modified, also needs to be set in inputfile then.

    qm_mm_boundary_dict = {}
    for qmatom in qmatoms:
        # Option below to skip creating boundaryatom pair (and subsequent linkatoms) if atom index is flagged
        # Applies to rare case where QM atom is bonded to MM atom but we don't want a linkatom.
        # Example: bridging sulfide in Cys that connects to Fe4S4 and H-cluster.
        if qmatom in excludeboundaryatomlist:
            logger.info(f"QMatom : {qmatom} in excludeboundaryatomlist: {excludeboundaryatomlist}")
            logger.info("Skipping QM-MM boundary...")
            continue
        # Note: get_connected_atoms very slow
        connatoms = get_connected_atoms_np(coords, elems, scale, tol, qmatom)
        # Find connected atoms that are not in QM-atoms
        boundaryatom = listdiff(connatoms, qmatoms)

        if len(boundaryatom) > 1:
            logger.error(f"Found more than 1 boundaryatom for QM-atom {qmatom} . This is considered unusual")
            logger.info(
                "This typically either happens when your QM-region is badly defined or a QM-atom is clashing with an "
                "MM atom"
            )
            logger.info("QM atom :  %s", qmatom)
            logger.info("MM Boundaryatoms (connected to QM-atom based on distance) :  %s", boundaryatom)
            logger.info("MM Boundary atom coordinates (for debugging):")
            for b in boundaryatom:
                logger.info(f"{b} {elems[b]} {coords[b][0]} {coords[b][1]} {coords[b][2]}")
            # Adding to dict
            qm_mm_boundary_dict[qmatom] = boundaryatom
        elif len(boundaryatom) == 1:
            # Warn if QM-MM boundary is not a plain-vanilla C-C bond
            if elems[qmatom] != "C" or elems[boundaryatom[0]] != "C":
                logger.warning("QM-MM boundary is not the ideal C-C scenario:")
                logger.warning(
                    f"QM-MM boundary: {elems[qmatom]}({qmatom}) - {elems[boundaryatom[0]]}({boundaryatom[0]})"
                )
                if unusualboundary is False:
                    raise InputError(
                        "Make sure you know what you are doing (note that atoms are counted from 0, not 1). "
                        "Exiting.\nTo override exit, add: unusualboundary=True  to QMMMTheory object"
                    )
            # Adding to dict
            qm_mm_boundary_dict[qmatom] = [boundaryatom[0]]
    logger.info("QM-MM boundary dictionary: %s", qm_mm_boundary_dict)
    log_time_since(timeA, "get_boundary_atoms")
    return qm_mm_boundary_dict


# Get linkatom positions for a list of qmatoms and the current set of coordinates
# Two methods: simple method (default) and ratio method.
# Simple method: Just use a fixed distance (default 1.09 Å)
# Ratio method: Determine by scaling QM1-MM1 distance with a ratio. Ratio can be fixed value (e.g. 0.723) or determined
# from equilibrium distances (not ready)
# Using linkatom distance of 1.09 Å for now as default. Makes sense for C-H link atoms.
def get_linkatom_positions(
    qm_mm_boundary_dict,
    qmatoms,
    coords,
    elems,
    linkatom_method="simple",
    linkatom_type="H",
    linkatom_simple_distance=None,
    bondpairs_eq_dict=None,
    linkatom_ratio=0.723,
):
    logger.info("Inside get_linkatom_positions")
    logger.info("linkatom_type: %s", linkatom_type)
    logger.info("linkatom_method: %s", linkatom_method)

    if linkatom_simple_distance is None:
        logger.info("linkatom_simple_distance not set. Getting standard distance from dictionary for each element:")
    else:
        logger.info("linkatom_simple_distance was set by user: %s", linkatom_simple_distance)
    # Dict of linkatom distances for different elements
    linkdistances_dict = {("C", "H"): 1.09, ("O", "H"): 0.98, ("N", "H"): 0.99}
    logger.info("Linkatom distance dictionary: %s", linkdistances_dict)
    # If dictionary of linkatom-distances provided then use that instead
    if linkatom_method == "ratio" and linkatom_ratio == "Auto" and bondpairs_eq_dict is None:
        # TODO: Determine automatically somehow
        bondpairs_eq_dict = {
            ("C", "H"): 1.09,
            ("C", "C"): 1.522269,
            ("C", "N"): 1.47,
            ("C", "O"): 1.43,
            ("C", "S"): 1.81,
        }

    # Get boundary atoms
    logger.info("qm_mm_boundary_dict: %s", qm_mm_boundary_dict)
    # Get coordinates for QMX and MMX pair. Create new L coordinate that has a modified distance to QMX
    linkatoms_dict = {}
    # Looping over QM-MM boundaries
    for dict_item in qm_mm_boundary_dict.items():
        qmatom = dict_item[0]
        # Looping over MM-atoms in boundary (i.e. we can have a MM1-QM1-MM1 situation e.g. requiring multiple linkatoms)
        for mmatom in dict_item[1]:
            qmatom_coords = np.array(coords[qmatom])
            mmatom_coords = np.array(coords[mmatom])
            # Determine linkatom distance
            if linkatom_method == "ratio":
                if linkatom_ratio == "Auto":
                    logger.info("Automatic ratio. Determining ratio based on dict of equilibrium distances")
                    # TODO
                    R_eq_QM_H = bondpairs_eq_dict[(elems[qmatom], linkatom_type)]
                    R_eq_QM_MM = bondpairs_eq_dict[(elems[qmatom], elems[mmatom])]
                    logger.info("R_eq_QM_H: %s", R_eq_QM_H)
                    logger.info("R_eq_QM_MM: %s", R_eq_QM_MM)
                    linkatom_ratio = R_eq_QM_H / R_eq_QM_MM
                    raise InputError(f"Determined ratio: {linkatom_ratio}\nnot yet ready")
                distance(qmatom_coords, mmatom_coords)
                # See https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9314059/
                linkatom_coords = linkatom_ratio * (mmatom_coords - qmatom_coords) + qmatom_coords
                linkatom_distance = distance(qmatom_coords, linkatom_coords)
                logger.info(
                    f"Linkatom distance (QM1-L) determined to be: {linkatom_distance} (using ratio {linkatom_ratio})"
                )
            elif linkatom_method == "simple":
                if linkatom_simple_distance is None:
                    # Getting from dict
                    linkatom_distance = linkdistances_dict[(elems[qmatom], linkatom_type)]
                else:
                    # Getting from user
                    linkatom_distance = linkatom_simple_distance
                logger.info("Linkatom distance (QM1-L) is: %s", linkatom_distance)
                # Determining coords
                linkatom_coords = list(
                    qmatom_coords
                    + (mmatom_coords - qmatom_coords) * (linkatom_distance / distance(qmatom_coords, mmatom_coords))
                )
            else:
                raise InputError("Invalid linkatom_method. Exiting.")

            linkatoms_dict[(qmatom, mmatom)] = linkatom_coords
    return linkatoms_dict


# Grabbing molecules from multi-XYZ trajectory file (can be MD-file, optimization traj etc).
# Creating fragments for each conformer
def get_molecules_from_trajectory(file, writexyz=False, skipindex=1, conncalc=False) -> list:
    """Create a Fragment for every snapshot in a multi-molecule XYZ trajectory.

    Args:
        file: path to the multi-molecule XYZ file.
        writexyz: also write each snapshot to its own XYZ file.
        skipindex: keep only every Nth snapshot.
        conncalc: calculate connectivity for each fragment (slow for large systems).

    Returns:
        List of Fragment objects labelled "<file>_<index>".
    """
    logger.info(small_header("Get molecules from trajectory"))
    logger.info("Finding molecules/snapshots in multi-XYZ trajectory file and creating fragments...")
    logger.info(f"Taking every {skipindex}th entry")
    list_of_molecules = []
    all_elems, all_coords, _all_titles = split_multimolxyzfile(
        file, writexyz=writexyz, skipindex=skipindex, return_fragments=False
    )
    logger.info(f"Found {len(all_elems)} molecules in file.")
    for i, (els, cs) in enumerate(zip(all_elems, all_coords, strict=False)):
        conf = openmmqmmm.Fragment(elems=els, coords=cs, conncalc=conncalc, label=f"{file}_{i}")
        list_of_molecules.append(conf)

    return list_of_molecules


# Get list of lists of water constraints in system (O-H,O-H,H-H) via OpenMM theory
def get_water_constraints(openmmtheoryobject=None, atomlist=None, watermodel="tip3p") -> list:
    """Return bond constraints for every water molecule in an OpenMM system.

    Water residues are identified by residue name (HOH, WAT, TIP) using the
    residue and element information stored on the OpenMMTheory object.

    Args:
        openmmtheoryobject: OpenMMTheory providing resnames and mm_elements.
        atomlist: atom indices to search (e.g. the active region).
        watermodel: water model name; "tip3p" and "spc" are supported.

    Returns:
        List of [i, j] atom-index pairs to constrain (O-H and H-H).

    Raises:
        InputError: for a missing argument, an unknown water model, or an
            OpenMMTheory object without residue/element information.
    """
    logger.info("Inside getwaterconstraintslist")
    if openmmtheoryobject is None or atomlist is None:
        raise InputError("getwaterconstraintslist requires openmmtheoryobject and atomlist to be set")
    if watermodel in {"tip3p", "spc"}:
        water_resname = ["HOH", "WAT", "TIP"]
    else:
        raise InputError("unknown watermodel")

    resnames = openmmtheoryobject.resnames
    elements = openmmtheoryobject.mm_elements

    if len(resnames) == 0:
        raise InputError("Error: No resnames found in OpenMMTheory object")
    if len(elements) == 0:
        raise InputError("Error: No mm_elements found in OpenMMTheory object")

    waterconstraints = []
    if resnames:
        for index, (rn, el) in enumerate(zip(resnames, elements, strict=False)):
            # Skipping if not in atomlist
            if index not in atomlist:
                continue

            if rn in water_resname and el == "O":
                waterconstraints.append([index, index + 1])
                waterconstraints.append([index, index + 2])
                waterconstraints.append([index + 1, index + 2])
    # if len(atomtypes) == 0:
    #    #NOTE: Atomtypes only defined if OpenMMTheory created from CHARMM Files
    #    # Assuming OT or OW oxygen atomtypes used if TIP3P. Assuming oxygen comes first
    #    # TODO: support more water models here. like 4-site and 5-site models
    #
    #    for index, at in enumerate(atomtypes):
    #        # Skipping if not in actatomslist
    #        if actatoms is not None:
    #            if index not in actatoms:
    #        if at in oxygenlabels:

    return waterconstraints


# Check if charge/mult variables are not None. If None check fragment
# Only done for QM theories not MM. Passing theorytype string (e.g. from theory.theorytype if available)
def check_charge_mult(charge, mult, theorytype, fragment, jobtype, theory=None):
    # Check if QM or QM/MM theory
    if theorytype == "QM":
        if charge is None or mult is None:
            logger.warning(f"Charge/mult was not provided to {jobtype}")
            if fragment.charge is not None and fragment.mult is not None:
                logger.warning(
                    f"Fragment contains charge/mult information: Charge: {fragment.charge} Mult: {fragment.mult}  "
                    f"Using this."
                )
                charge = fragment.charge
                mult = fragment.mult
            else:
                raise InputError("No charge/mult information present in fragment either. Exiting.")
    elif theorytype == "QM/MM":
        # Note: theory needs to be set
        if charge is None or mult is None:
            logger.warning(f"Charge/mult was not provided to {jobtype}")
            logger.info("Checking if present in QM/MM object")
            if theory.qm_charge is not None and theory.qm_mult is not None:
                charge = theory.qm_charge
                mult = theory.qm_mult
                logger.info("Found qm_charge and qm_mult attributes.")
                logger.info(f"Using charge={charge} and mult={mult}")
            elif fragment.charge is not None and fragment.mult is not None:
                logger.warning(
                    f"Fragment contains charge/mult information: Charge: {fragment.charge} Mult: {fragment.mult} Using "
                    f"this instead"
                )
                logger.warning("Make sure this is what you want!")
                charge = fragment.charge
                mult = fragment.mult
            else:
                raise InputError("No charge/mult information present in fragment either. Exiting.")
    elif theorytype == "ONIOM":
        logger.info("Checking if charge/mult information present in ONIOM object")
        if theory.fullregion_charge is not None and theory.fullregion_mult is not None:
            logger.info("Found fullregion_charge and fullregion_mult attributes.")
            logger.info("All good, continuing\n")
    elif theorytype == "MM":
        # Setting charge/mult to None if MM
        charge = None
        mult = None
    return charge, mult


# Get list of bad atoms based on supplied fragment and gradient
def check_gradient_for_bad_atoms(fragment=None, gradient=None, threshold=45000) -> list[int]:
    """Report atoms with unusually large gradient components (useful for spotting clashes)."""
    indices = []
    logger.info("Checking system total gradient for bad atoms")
    logger.info("Gradient threshold setting: %s", threshold)
    for i, k in enumerate(gradient):
        if any(abs(k) > threshold):
            indices.append(i)
    if len(indices) > 0:
        logger.info("The following atoms have abnormally high values, probably due to bad atom positions:")
        logger.info("")
        logger.info("Index    Element           Coordinates                              Gradient")
        for i in indices:
            logger.info(
                f"{i:7} {fragment.elems[i]:>5} {fragment.coords[i][0]:>12.6f} {fragment.coords[i][1]:>12.6f} "
                f"{fragment.coords[i][2]:>12.6f}      {gradient[i][0]:>6.3f} {gradient[i][1]:>6.3f} "
                f"{gradient[i][2]:>6.3f}"
            )
        logger.info("")
        logger.info(
            "These atoms may need to be constrained (e.g. if metal-cofactor) or atom positions need to be corrected "
            "before starting simulation"
        )
    else:
        logger.info("")
        logger.info(f"No atoms with gradients larger than threshold: {threshold}")
    return indices


# Define XH bond constraints for a given fragment and a set of atomindices (e.g. an active region)
# and an optional exclusion list (e.g. QM-region)
def define_xh_constraints(fragment, actatoms=None, excludeatoms=None) -> list:
    """Return X-H bond constraints for a fragment or a region of it.

    Every hydrogen is paired with the atom it is bonded to, as determined from
    the calculated connectivity.

    Args:
        fragment: Fragment holding elements and coordinates.
        actatoms: atom indices to restrict the search to (the active region);
            all atoms if None. Returned indices are always full-system indices.
        excludeatoms: atom indices to leave unconstrained.

    Returns:
        List of [X, H] atom-index pairs to constrain.

    Raises:
        InternalError: if a hydrogen is not found bonded to exactly one atom.
    """
    logger.info("Inside define_XH_constraints function")
    if actatoms is None:
        subset_elems = fragment.elems
        subset_coords = fragment.coords
        actatoms = fragment.atomlist
    else:
        subset_elems = [fragment.elems[i] for i in actatoms]
        subset_coords = np.take(fragment.coords, actatoms, axis=0)

    logger.info(f"Defining constraints for {len(subset_elems)} atom-region")

    # Finding H-atoms (both act indices and full indices)
    tempHatoms = [index for index, el in enumerate(subset_elems) if el == "H"]
    tempHatoms_full = [actindex_to_fullindex(i, actatoms) for i in tempHatoms]
    Hatoms = []
    if excludeatoms is not None:
        logger.info("Checking for exclude atoms")
        for th, th_f in zip(tempHatoms, tempHatoms_full, strict=False):
            if th_f not in excludeatoms:
                Hatoms.append(th)
    else:
        Hatoms = tempHatoms

    # Now finding X-H pairs for active region
    # py version (slow) but good enough for a few thousand atoms
    scale = CONNECTIVITY_SCALE
    tol = CONNECTIVITY_TOL
    act_con_list = []
    for Hatom in Hatoms:
        connatoms = get_connected_atoms_np(subset_coords, subset_elems, scale, tol, Hatom)
        act_con_list.append(connatoms)
    # Convert XH actregion indices to finalregion indices
    final_list = []
    for XHpair in act_con_list:
        if len(XHpair) != 2:
            raise InternalError(f"XHpair is strange: {XHpair}")
        final_list.append([actindex_to_fullindex(XHpair[0], actatoms), actindex_to_fullindex(XHpair[1], actatoms)])
    return final_list


# Simple function to convert atom indices from full system to Active region. Single index case
def fullindex_to_actindex(fullindex, actatoms):
    return actatoms.index(fullindex)


# Simple function to convert atom indices from active region to full-system case.
def actindex_to_fullindex(actindex, actatoms):
    return actatoms[actindex]


# Simple get_water constraints for fragment without doing connectivity
# Limitation: Assumes all waters from starting index to end and that waters are ordered: O H H
def simple_get_water_constraints(fragment, starting_index=None, onlyHH=False) -> list:
    """Return water bond constraints by position, without residue information.

    Assumes the water molecules are stored in O, H, H order and occupy a
    contiguous block at the end of the coordinate file.

    Args:
        fragment: Fragment holding the elements.
        starting_index: index of the oxygen of the first water molecule.
        onlyHH: constrain only the H-H distance, leaving the O-H bonds free.

    Returns:
        List of [i, j] atom-index pairs to constrain.

    Raises:
        InputError: if starting_index is missing or does not point at an oxygen.
    """
    logger.info("Inside simple_get_water_constraints function")
    logger.warning(
        "Warning: Note that water residues have to have O,H,H order and have to be at the end of the coordinate file"
    )
    logger.info("Starting index for first water oxygen: %s", starting_index)
    if starting_index is None:
        raise InputError("Error: You must provide a starting_index value!")
    if fragment.elems[starting_index] != "O":
        raise InputError(
            (
                "Starting atom for water fragment is not oxygen!\n{}\n"
                "Also note that water fragments must have O H H order!"
            ).format(f"Make sure starting index ({starting_index}) is correct")
        )
    if onlyHH is False:
        logger.info("onlyHH is False. Will create list of O-H1, O-H2 and H1-H2 constraints")
    elif onlyHH is True:
        logger.info("onlyHH is True. Will create list of H1-H2 constraints only")
    constraints = []
    for i in range(starting_index, fragment.numatoms):
        if fragment.elems[i] == "O":
            # X-H constraint
            if onlyHH is False:
                constraints.append([i, i + 1])
                constraints.append([i, i + 2])
            # H-H constraints. i.e. effectively freezing angles
            constraints.append([i + 1, i + 2])
    return constraints


# Combien and place 2 fragments
def combine_and_place_fragments(ref_frag, trans_frag):
    for displacement in [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0]:
        # Translating 2nd frag by displacement in
        trans_frag.coords[:, -1] += displacement
        members = get_molecule_members_loop_np2(
            np.vstack((ref_frag.coords, trans_frag.coords)),
            ref_frag.elems + trans_frag.elems,
            10,
            1.0,
            0.4,
            atomindex=0,
            membs=None,
        )
        if len(members) == ref_frag.numatoms:
            logger.info("Molecules are sufficiently far apart")
            break

    return Fragment(elems=ref_frag.elems + trans_frag.elems, coords=np.vstack((ref_frag.coords, trans_frag.coords)))


# Simple function to combine 2 fragments where one is assumed to be a solute (fewer atoms) and the other assumed to be
# some kind of solvent system (box,sphere etc.)
# Use tolerance (tol) e.g. to control how many solvent molecules around get deleted
# Currently using 0.4 as default based on threonine in acetonitrile example
def insert_solute_into_solvent(
    solute=None,
    solute2=None,
    solvent=None,
    scale=1.0,
    tol=0.4,
    write_pdb=False,
    write_solute_connectivity=True,
    solute_pdb=None,
    solute2_pdb=None,
    solvent_pdb=None,
    outputname="solution.pdb",
    write_pbc_info=True,
) -> "Fragment":
    """Insert one or two solute molecules into a solvent box, removing clashes.

    Solvent molecules overlapping the solute are deleted whole, so the result
    stays chemically sensible.

    Args:
        solute: Fragment for the solute.
        solute2: optional second solute Fragment.
        solvent: Fragment holding the pre-equilibrated solvent box.
        scale: covalent-radius scale factor used for clash detection.
        tol: covalent-radius tolerance (Angstrom) used for clash detection.
        write_pdb: also write the combined system to a PDB file.
        write_solute_connectivity: include CONECT records for the solute.
        solute_pdb: PDB file supplying the topology metadata for the solute.
        solute2_pdb: PDB file supplying the topology metadata for the second solute.
        solvent_pdb: PDB file supplying the topology metadata for the solvent.
        outputname: filename for the written PDB file.
        write_pbc_info: write the solvent cell dimensions as a CRYST1 record.

    Returns:
        A new Fragment containing the solute(s) plus the trimmed solvent.
    """
    logger.info("\ninsert_solute_into_solvent\n")
    # Early exits
    if write_pdb:
        logger.info("Write PDB option is active.")
        if solute_pdb is None or solvent_pdb is None:
            raise InputError("Error: write_pdb is active but no input solute_pdb or solvent_pdb files were provided")
    if solute is None and solute_pdb is not None:
        logger.info("No solute fragment provided but solute_pdb is set. Reading solute fragment from PDB-file")
        solute = Fragment(pdbfile=solute_pdb)
    if solute2 is None and solute2_pdb is not None:
        logger.info("No solute2 fragment provided but solute2_pdb is set. Reading solute2 fragment from PDB-file")
        solute2 = Fragment(pdbfile=solute2_pdb)
    if solvent is None and solvent_pdb is not None:
        logger.info("No solvent fragment provided but solvent_pdb is set. Reading solvent fragment from PDB-file")
        solvent = Fragment(pdbfile=solvent_pdb)

    # Get centers
    com_box = solvent.get_coordinate_center()

    if solute2 is None:
        com_solute = solute.get_coordinate_center()
        # Translating solute coords
        trans_coord = np.array(com_box) - np.array(com_solute)
        solute.coords = solute.coords + trans_coord
    else:
        # Combine and Translate solute2 so that it is close to solute1
        combined_solute = combine_and_place_fragments(ref_frag=solute, trans_frag=solute2)

        # COM and trans
        com_solute = combined_solute.get_coordinate_center()
        trans_coord = np.array(com_box) - np.array(com_solute)
        # Translate combined solute fragment
        combined_solute.coords = combined_solute.coords + trans_coord

    # New Fragment by combining element lists and coordinates
    if solute2 is None:
        logger.info("Combining solute and solvent")
        new_frag = Fragment(elems=solute.elems + solvent.elems, coords=np.vstack((solute.coords, solvent.coords)))
    else:
        logger.info("Combining combined_solute and solvent")
        new_frag = Fragment(
            elems=combined_solute.elems + solvent.elems,
            coords=np.vstack((combined_solute.coords, solvent.coords)),
        )

    new_frag.write_xyzfile(xyzfilename="solution-pre.xyz")
    # Trim by removing clashing atoms

    # Find atoms connected to solute index 0. Uses scale and tol
    membs = get_molecule_members_loop_np2(new_frag.coords, new_frag.elems, 20, scale, tol, atomindex=0, membs=None)
    delatoms = [i for i in membs if i >= solute.numatoms]
    logger.info("First delatoms: %s", delatoms)
    if solute2 is not None:
        membs2 = get_molecule_members_loop_np2(
            new_frag.coords, new_frag.elems, 20, scale, tol, atomindex=solute.numatoms, membs=None
        )
        logger.info("membs2: %s", membs2)
        for j in membs2:
            if j >= solute.numatoms + solute2.numatoms and j not in delatoms:
                delatoms.append(j)
    logger.info("Final delatoms: %s", delatoms)

    # Deleting
    delatoms.sort(reverse=True)
    logger.info("Found clashing solvent atoms: %s", delatoms)
    for d in delatoms:
        new_frag.delete_atom(d)
    logger.info("")
    logger.info("Final fragment after removing clashing atoms:")
    new_frag.update_attributes()
    new_frag.write_xyzfile(xyzfilename="solution.xyz")

    # WRITE PDB
    if write_pdb:
        logger.info("Write_PDB is active. Will write PDB-file of solute+solvent system for topology purposes")
        try:
            import openmm.app
        except ImportError:
            raise MissingDependencyError("Error: OpenMM library not found. Please install OpenMM") from None

        # PDB-files
        pdb1 = openmm.app.PDBFile(solute_pdb)
        solute_resname = next(iter(pdb1.topology.residues())).name
        logger.info("solute_resname: %s", solute_resname)
        pdb2 = openmm.app.PDBFile(solvent_pdb)
        solvent_box_vectors = pdb2.topology.getPeriodicBoxVectors()
        logger.info("Found PBC vectors in solvent PDB-file: %s", solvent_box_vectors)

        # Create modeller object
        modeller = openmm.app.Modeller(pdb1.topology, pdb1.positions)  # Add pdbfile1

        # solute2
        if solute2 is not None:
            logger.info("Adding solute2")
            pdb_solute2 = openmm.app.PDBFile(solute2_pdb)
            solute2_resname = next(iter(pdb_solute2.topology.residues())).name
            logger.info("solute2_resname: %s", solute2_resname)
            modeller.add(pdb_solute2.topology, pdb_solute2.positions)  # Add pdbfile2
        logger.info("Adding solvent")
        modeller.add(pdb2.topology, pdb2.positions)  # Add pdbfile2

        # Delete clashing atoms from topology
        toDelete = [r for j, r in enumerate(modeller.topology.atoms()) if j in delatoms]
        modeller.delete(toDelete)
        mergedPositions = new_frag.coords

        # Delete solute connectivity if chosen so not printed in PDB
        if write_solute_connectivity is True:
            logger.info(
                "Will write solute connectivity to PDB-file. Necessary for OpenMM topology recognition when bonded MM "
                "parameters are used."
            )
        else:
            logger.info(
                "Will NOT write solute connectivity to PDB-file. Necessary for OpenMM topology recognition when bonded "
                "MM parameters are NOT used."
            )
            logger.info("Num bonds in topology: %s", modeller.topology.getNumBonds())
            solute_bonds = [i for i in modeller.topology.bonds() if i[0].residue.name == solute_resname]
            logger.info("Solute bonds: %s", solute_bonds)
            logger.info("Deleting solute bonds")
            modeller.delete(solute_bonds)
            logger.info("Num bonds in topology: %s", modeller.topology.getNumBonds())

        # PBC info
        if write_pbc_info:
            logger.info("write_PBC_info True: Writing PBC to header of PDB-file")
            if solvent_box_vectors is not None:
                logger.info("PBC vectors found in solvent PDB-file: %s", solvent_box_vectors)
                logger.info("Adding to solution PDB-file")
                modeller.topology.setPeriodicBoxVectors(solvent_box_vectors)

        # Write merged topology and positions to new PDB file
        openmmqmmm.openmm.write_pdbfile_openmm_topology(modeller.topology, mergedPositions, outputname)
    return new_frag


# Basic fast function to calculate the Coulomb energy.
# Assumes coords in Angstrom
def nuc_nuc_repulsion(coords, charges) -> float:
    """Return the classical nucleus-nucleus repulsion energy of a set of point charges.

    Args:
        coords: (natoms, 3) array of coordinates in Angstrom.
        charges: per-atom charges (nuclear charges for the usual use).

    Returns:
        Repulsion energy in Hartree.
    """
    charges = np.array(charges)  # Ensure charges is a numpy array
    coords_b = coords * 1.88972612546
    diff = coords_b[:, None, :] - coords_b[None, :, :]
    distances = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(distances, np.inf)
    return 0.5 * np.sum(charges[:, None] * charges[None, :] / distances)


# Very simple dummy topology (no connectivity or bonds)
def define_dummy_topology(elems, scale=1.0, tol=0.1, resname="MOL"):
    try:
        import openmm.app
    except ImportError:
        raise InputError("Error: OpenMM not found. Cannot define a topology") from None
    logger.info("Defining new basic single-chain, multi-residue topology")
    pdb_topology = openmm.app.Topology()
    chain = pdb_topology.addChain()
    # Looping over molecules defined by connectivity
    residue = pdb_topology.addResidue(resname, chain)

    # Defaultdictionary to keep track of unique element-atomnames
    atomnames_dict = defaultdict(int)
    for el in elems:
        atomnumber = openmm.app.Element.getBySymbol(el).atomic_number
        element = openmm.app.Element.getByAtomicNumber(atomnumber)
        # Define unique atomname
        atomnames_dict[el] += 1
        atomname = f"{el}{atomnames_dict[el]}"
        pdb_topology.addAtom(atomname, element, residue)
    return pdb_topology

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

CONNECTIVITY_SCALE = 1.0
CONNECTIVITY_TOL = 0.1


class Reaction:
    """A reaction: an ordered list of fragments with stoichiometry (and optional energies)."""

    def __init__(self, fragments, stoichiometry, label=None, unit="eV"):
        logger.info(sub_header("New reaction"))

        self.fragments = fragments
        self.stoichiometry = stoichiometry
        self.check_fragments()
        self.elements = [item for sublist in [frag.elems for frag in fragments] for item in sublist]

        self.label = label

        self.unit = unit

        self.energies = []
        self.reaction_energy = None

        # Keeping track of orbital-files: key: 'SCF':["frag1.gbw","frag2.gbw","frag3.gbw"],
        # 'MP2nat':["frag1.gbw","frag2.gbw","frag3.gbw"]
        self.orbital_dictionary = defaultdict(list)
        self.properties = defaultdict(list)

    def reset_energies(self):
        """Discard the stored fragment energies and reaction energy."""
        self.energies = []
        self.reaction_energy = None

    def check_fragments(self):
        """Validate the reaction definition."""
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
        """Combine the stored fragment energies into the reaction energy."""
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


class Fragment:
    """Molecular system: elements, coordinates, charge/multiplicity, connectivity and topology."""

    def __init__(
        self,
        *,
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
    ):
        self.charge = None
        self.mult = None

        self.label = None

        logger.info(sub_header("New fragment"))
        logger.info("Fragment creation")
        self.energy = None
        self.elems = []
        self.coords = np.zeros((0, 3))
        self.connectivity = []
        self.atomcharges = []
        self.atomtypes = []
        self.pdb_atomnames = None
        self.pdb_resnames = None
        self.pdb_chainlabels = None
        self.pdb_residlabels = None
        self.pdb_conect_lines = None
        self.pdb_topology = None  # New, use OpenMM to read PDB-file and get topology
        self.Centralmainfrag = []
        self.formula = None
        if atomcharges is not None:
            self.atomcharges = atomcharges
        if atomtypes is not None:
            self.atomtypes = atomtypes
        # Hessian. Can be added by Numfreq/Anfreq job
        self.hessian = None

        self.fragmenttype_labels = []

        if coords is not None:
            self.coords = _reformat_list_to_array(coords)
            if elems is None:
                raise InputError("Error: Coords list provided but no elems list. Exiting.")
            if len(elems) != len(coords):
                raise InputError(
                    f"Error: Coords list (len {len(coords)}) and elems list ({len(elems)}) have different lengths. "
                    f"Exiting."
                )
            self.elems = elems
            if connectivity is not None:
                conncalc = False
                self.connectivity = connectivity
        elif fragments is not None:
            logger.info("Creating fragments by combining input fragments")
            self.elems = []
            for f in fragments:
                self.elems += f.elems
            self.coords = np.vstack([f.coords for f in fragments])

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
        elif atom is not None:
            logger.info("Creating Atom Fragment")
            self.elems = [atom]
            self.coords = _reformat_list_to_array([[0.0, 0.0, 0.0]])
        elif diatomic is not None:
            logger.info("Creating Diatomic Fragment from formula and bondlength")
            if bondlength is None:
                if diatomic_bondlength is None:
                    raise InputError("diatomic option requires bondlength to be set. Exiting!")
                bondlength = diatomic_bondlength
            self.elems = _formula_to_elem_list(diatomic)
            if len(self.elems) != 2:
                raise InputError(f"Problem with molecular formula diatomic={diatomic} string!")
            self.coords = _reformat_list_to_array([[0.0, 0.0, 0.0], [0.0, 0.0, float(bondlength)]])
        elif coordsstring is not None:
            self.add_coords_from_string(coordsstring, scale=scale, tol=tol, conncalc=conncalc)
        elif smiles is not None:
            self.create_coords_from_smiles(smiles)
        elif xyzfile is not None:
            if not os.path.isfile(xyzfile):
                raise InputError(f"XYZ-file {xyzfile} not found. Exiting.")

            self.label = xyzfile.split("/")[-1].split(".")[0]
            self.read_xyzfile(xyzfile, readchargemult=readchargemult, conncalc=conncalc)
        elif pdbfile is not None:
            self.label = pdbfile.split("/")[-1].split(".")[0]
            self.read_pdbfile_openmm(pdbfile)
        elif pdbxfile is not None:
            self.label = pdbxfile.split("/")[-1].split(".")[0]
            self.read_pdbxfile(pdbxfile)
        elif grofile is not None:
            self.label = grofile.split("/")[-1].split(".")[0]
            self.read_grofile(grofile, conncalc=False)
        elif amber_inpcrdfile is not None:
            self.label = amber_inpcrdfile.split("/")[-1].split(".")[0]
            logger.info("Reading Amber INPCRD file")
            if amber_prmtopfile is None:
                raise InputError("amber_prmtopfile argument must be provided as well!")
            self.read_amberfile(inpcrdfile=amber_inpcrdfile, prmtopfile=amber_prmtopfile, conncalc=conncalc)
        elif chemshellfile is not None:
            self.label = chemshellfile.split("/")[-1].split(".")[0]
            self.read_chemshellfile(chemshellfile, conncalc=conncalc)
        elif fragfile is not None:
            self.label = fragfile.split("/")[-1].split(".")[0]
            self.read_fragment_from_file(fragfile)
        else:
            raise InputError("Fragment requires some kind of valid coordinate input!")
        if label is not None:
            self.label = label

        if charge is not None:
            self.charge = charge
        if mult is not None:
            self.mult = mult

        self.update_attributes()
        if conncalc is True and len(self.connectivity) == 0:
            self.calc_connectivity(scale=scale, tol=tol)

        # Constraints attributes. Used by parallel surface-scan to pass constraints along.
        # Populated by calc_surface relaxed para
        self.constraints = None

    def __repr__(self):
        label = f" {self.label!r}" if self.label is not None else ""
        chargemult = f"charge={self.charge} mult={self.mult}"
        return f"<Fragment{label}: {self.prettyformula}, {self.numatoms} atoms, {chargemult}>"

    __str__ = __repr__

    def info(self):
        """Log a summary of the fragment: formula, atom count, charge and multiplicity."""
        logger.info("Fragment object")
        logger.info("%s", self.__dict__)

    def update_attributes(self):
        """Recompute the derived attributes after the coordinates or elements change."""
        logger.info("Creating/Updating fragment attributes...")
        if len(self.coords) == 0:
            raise InputError("No coordinates in fragment. Something went wrong. Exiting.")
        if not isinstance(self.coords, np.ndarray):
            raise InputError("self.coords is not a numpy array. Something is wrong. Exiting.")
        self.nuccharge = total_nuclear_charge(self.elems)
        self.nuc_charges = elems_to_nuclear_charges(self.elems)
        self.numatoms = len(self.coords)
        self.atomlist = list(range(self.numatoms))
        self.allatoms = self.atomlist
        self.mass = total_mass(self.elems)
        self.list_of_masses = list_of_masses(self.elems)
        self.masses = self.list_of_masses
        self.formula = elems_to_formula(self.elems)
        self.prettyformula = self.formula
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

    def add_coords_from_string(self, coordsstring, scale=None, tol=None, conncalc=False):
        """Append atoms parsed from a multi-line "El x y z" coordinate string."""
        logger.info("Getting coordinates from string: %s", coordsstring)
        if len(self.coords) > 0:
            logger.info("Fragment already contains coordinates")
            logger.info("Adding extra coordinates")
        coordslist = coordsstring.split("\n")
        tempcoords = []
        for line in coordslist:
            if len(line) > 5:
                self.elems.append(reformat_element(line.split()[0]))
                clist = [float(line.split()[1]), float(line.split()[2]), float(line.split()[3])]
                tempcoords.append(clist)
        self.coords = _reformat_list_to_array(tempcoords)
        self.label = "".join(self.elems)

    def create_coords_from_smiles(self, smiles):
        """Generate 3D coordinates from a SMILES string (requires OpenBabel)."""
        logger.info("Creating coordinates from SMILES string: %s", smiles)
        from openmmqmmm.openbabel import smiles_to_coords

        elems, coords = smiles_to_coords(smiles)
        self.elems = elems
        self.coords = _reformat_list_to_array(coords)
        self.update_attributes()

    def replace_coords(self, elems, coords, conn=False, scale=None, tol=None):
        """Replace the elements and coordinates with a new set."""
        logger.info("Replacing coordinates in fragment.")

        self.elems = elems
        self.coords = _reformat_list_to_array(coords)
        self.update_attributes()
        if conn is True:
            self.calc_connectivity(scale=scale, tol=tol)

    def get_non_h_atomindices(self):
        """Return the indices of all atoms that are not hydrogen."""
        return [index for index, el in enumerate(self.elems) if el != "H"]

    def delete_atom(self, atomindex):
        """Remove one atom and refresh the derived attributes."""
        self.coords = np.delete(self.coords, atomindex, axis=0)
        self.elems.pop(atomindex)
        self.atomcharges.pop(atomindex)
        self.atomtypes.pop(atomindex)
        self.fragmenttype_labels.pop(atomindex)

        self.update_attributes()

    def print_coords(self):
        """Log the coordinates of every atom in the fragment."""
        logger.info("Cartesian coordinates (Å):")
        for i, (el, c) in enumerate(zip(self.elems, self.coords, strict=False)):
            line = f" {i:<4} {el:4} {c[0]:>12.6f} {c[1]:>12.6f} {c[2]:>12.6f}"
            logger.info("%s", line)

    def print_coords_for_atoms(self, members, labels=None):
        """Log the coordinates of selected atoms."""
        print_coords_for_atoms(self.coords, self.elems, members, labels=labels)

    def read_amberfile(self, inpcrdfile=None, prmtopfile=None, conncalc=False):
        """Read coordinates and topology from Amber inpcrd/prmtop files."""
        logger.info(
            f"Reading coordinates from Amber INPCRD file: '{inpcrdfile}' and PRMTOP file: '{prmtopfile}' into fragment."
        )
        try:
            elems, coords, _box_dims = read_ambercoordinates(prmtopfile=prmtopfile, inpcrdfile=inpcrdfile)
        except FileNotFoundError:
            raise FileFormatError(f"File {prmtopfile} or {inpcrdfile} not found") from None
        self.coords = _reformat_list_to_array(coords)
        self.elems = elems

    def read_grofile(self, filename, conncalc=False, scale=None, tol=None):
        """Read coordinates from a GROMACS .gro file."""
        logger.info(f"Reading coordinates from Gromacs GRO file '{filename}' into fragment")
        try:
            elems, coords, _boxdims = read_gromacsfile(filename)
        except FileNotFoundError:
            raise FileFormatError(f"File '{filename}' not found") from None
        self.coords = coords
        self.elems = elems

    def read_chemshellfile(self, filename, conncalc=False, scale=None, tol=None):
        """Read coordinates from a ChemShell fragment file (Bohr units)."""
        logger.info(f"Reading coordinates from Chemshell file '{filename}' into fragment.")
        try:
            elems, coords = _read_chemshellfragfile_xyz(filename)
        except FileNotFoundError:
            raise FileFormatError(f"File '{filename}' not found.") from None
        self.coords = coords
        self.elems = elems

    def read_pdbfile_openmm(self, filename):
        """Read a PDB file using OpenMM's parser, keeping the full topology."""
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

        self.pdb_topology = pdb.topology

    def read_pdbxfile(self, filename):
        """Read a PDBx/mmCIF file using OpenMM's parser, keeping the full topology."""
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

        self.pdb_topology = pdb.topology

    def read_xyzfile(self, filename, scale=None, tol=None, readchargemult=False, conncalc=True):
        """Read coordinates from an XYZ file."""
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
                    if isint(line.split()[0]) is True:
                        el = reformat_element(int(line.split()[0]), isatomnum=True)
                        self.elems.append(el)
                    else:
                        el = line.split()[0]
                        self.elems.append(reformat_element(el))
                    coords.append([float(line.split()[1]), float(line.split()[2]), float(line.split()[3])])
        self.coords = _reformat_list_to_array(coords)
        if self.numatoms != len(self.coords):
            raise FileFormatError("Number of atoms in header not equal to number of coordinate-lines. Check XYZ file!")

    def set_energy(self, energy):
        """Store a total energy on the fragment."""
        self.energy = float(energy)

    def get_coordinate_center(self):
        """Return the mean position of all atoms as a list (unweighted by mass)."""
        center_x = np.mean(self.coords[:, 0])
        center_y = np.mean(self.coords[:, 1])
        center_z = np.mean(self.coords[:, 2])
        return [center_x, center_y, center_z]

    # NOTE: This also returns elements, bit silly
    def get_coords_for_atoms(self, atoms):
        """Return the coordinates and elements of a subset of atoms."""
        subcoords = np.take(self.coords, atoms, axis=0)
        subelems = [self.elems[i] for i in atoms]
        return subcoords, subelems

    def calc_connectivity(self, conndepth=99, scale=None, tol=None):
        """Compute the connectivity table and store it on the fragment."""
        logger.info("Calculating connectivity.")
        if len(self.coords) > 10000:
            logger.info("Atom number > 10K. Connectivity calculation could take a while")

        if scale is None:
            scale = CONNECTIVITY_SCALE
            tol = CONNECTIVITY_TOL
        logger.info(f"Using scale: {scale} and tol: {tol} ")

        timestampA = time.time()
        fraglist = _calc_conn_py(self.coords, self.elems, conndepth, scale, tol)
        log_time_since(timestampA, "calc connectivity py")
        self.connectivity = fraglist
        conn_number_sum = 0
        for sublist in self.connectivity:
            conn_number_sum += len(sublist)
        if self.numatoms != conn_number_sum:
            raise InputError(
                f"Connectivity problem\nself.connectivity: {self.connectivity}\nconn_number_sum: "
                f"{conn_number_sum}\nself numatoms {self.numatoms}"
            )

    def get_centroid(self):
        """Return the mean position of all atoms as an array (unweighted by mass)."""
        return np.mean(self.coords, axis=0)

    def write_pdbfile(self, filename="Fragment"):
        """Write a PDB file using the fragment's own stored PDB information."""
        logger.info("Fragment.write_pdbfile method called")
        filename = filename.replace(".pdb", "")
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

    def define_topology(self, scale=1.0, tol=0.1, resname="MOL"):
        """Build an OpenMM topology for the fragment from its connectivity."""
        try:
            import openmm.app
        except ImportError:
            raise InputError("Error: OpenMM not found. Cannot define a topology") from None
        logger.info("Defining new basic single-chain, multi-residue topology")
        self.pdb_topology = openmm.app.Topology()
        chain = self.pdb_topology.addChain()

        if self.connectivity is None or (isinstance(self.connectivity, list) and len(self.connectivity) == 0):
            self.calc_connectivity(scale=scale, tol=tol)

        connectivity_dict = get_connected_atoms_dict(self.coords, self.elems, scale, tol)
        for mol in self.connectivity:
            logger.info("mol: %s", mol)
            residue = self.pdb_topology.addResidue(resname, chain)
            logger.info("residue: %s", residue)

            atomnames_dict = defaultdict(int)
            for at in mol:
                el = self.elems[at]
                atomnumber = openmm.app.Element.getBySymbol(el).atomic_number
                element = openmm.app.Element.getByAtomicNumber(atomnumber)
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

    def write_pdbfile_openmm(
        self, filename="Fragment", calc_connectivity=False, pdb_topology=None, skip_connectivity=False, resname="MOL"
    ):
        """Write a PDB file via OpenMM, building a topology if none is defined."""
        logger.info("write_pdbfile_openmm\n")
        try:
            import openmm.app
        except ImportError:
            raise InputError(
                "Error: OpenMM library not found. the OpenMM library is required to write PDB files."
            ) from None

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

        if skip_connectivity is True:
            logger.info("skip_connectivity True: this will not write connectivity lines to PDB-file")
            logger.info("Deleting molecule bond information")
            self.pdb_topology._bonds = []
        with open(filename, "w") as pdbhandle:
            openmm.app.PDBFile.writeFile(self.pdb_topology, self.coords, file=pdbhandle)
        logger.info(f"Wrote PDB-file: {filename}")
        return filename

    def write_xyzfile(
        self, xyzfilename="Fragment-xyzfile.xyz", writemode="w", write_chargemult=True, write_energy=True
    ):
        """Write the coordinates to an XYZ file."""
        with open(xyzfilename, writemode) as ofile:
            ofile.write(str(len(self.elems)) + "\n")
            if write_chargemult is True and write_energy is True:
                ofile.write(f"{self.charge} {self.mult} {self.energy}\n")
            else:
                ofile.write("title\n")

            for el, c in zip(self.elems, self.coords, strict=False):
                line = f"{el:4} {c[0]:14.8f} {c[1]:14.8f} {c[2]:14.8f}"
                ofile.write(line + "\n")
        logger.info("Wrote XYZ file:  %s", xyzfilename)
        return xyzfilename

    def write_xyz_for_atoms(self, xyzfilename="Fragment-subset.xyz", atoms=None):
        """Write an XYZ file containing only the selected atoms."""
        subset_elems = [self.elems[i] for i in atoms]
        subset_coords = np.take(self.coords, atoms, axis=0)
        with open(xyzfilename, "w") as ofile:
            ofile.write(str(len(subset_elems)) + "\n")
            ofile.write("title" + "\n")
            for el, c in zip(subset_elems, subset_coords, strict=False):
                line = f"{el:4} {c[0]:>12.6f} {c[1]:>12.6f} {c[2]:>12.6f}"
                ofile.write(line + "\n")

    def print_system(self, filename="fragment.frag"):
        """Write the full fragment (coordinates, charge, mult, connectivity) to a .frag file."""
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

    def read_fragment_from_file(self, fragfile):
        """Load a fragment previously written by print_system."""
        logger.info("Reading fragment from file: %s", fragfile)
        coordgrab = False
        coords = []
        elems = []
        atomcharges = []
        atomtypes = []
        fragment_type_labels = []
        connectivity = []
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
                    if "===============" in line:
                        coordgrab = False
                        continue
                    elems.append(line.split()[1])
                    coords.append([float(line.split()[2]), float(line.split()[3]), float(line.split()[4])])
                    atomcharges.append(float(line.split()[5]))
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
        self.coords = np.array(coords)
        self.atomcharges = atomcharges
        self.atomtypes = atomtypes
        self.fragmenttype_labels = fragment_type_labels
        self.update_attributes()
        self.connectivity = connectivity
        self.Centralmainfrag = Centralmainfrag


def _reformat_list_to_array(data):
    if isinstance(data, np.ndarray):
        return data
    if isinstance(data, list):
        if any(isinstance(el, list) for el in data) is False:
            raise InputError("Error (reformat_list_to_array): input should be a list of lists, not just a list")
        return np.array(data)
    raise InputError(
        "Error (reformat_list_to_array): coordinates must be a list of lists or a numpy array, "
        f"got {type(data).__name__}"
    )


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


def _print_internal_coordinate_table(fragment, actatoms=None):
    def _measure_bond(coords, i, j):
        return float(np.linalg.norm(coords[i] - coords[j]))

    def _measure_angle(coords, i, j, k):
        v1 = coords[i] - coords[j]
        v2 = coords[k] - coords[j]
        cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        return float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))

    def _measure_dihedral(coords, i, j, k, l):  # noqa: E741 - dihedral atoms i-j-k-l
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
        for j in conn[i]:
            bond_key = tuple(sorted((i, j)))
            if bond_key not in seen_bonds:
                val = _measure_bond(coords, i, j)
                label = f"{elems[i]}-{elems[j]}"
                logger.info(f"{'Bond':<10} {bond_key!s:<20} {label:<15} {val:>10.4f} Å")
                seen_bonds.add(bond_key)

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

        for j in conn[i]:
            for h in conn[i]:
                if h == j:
                    continue
                for k in conn[j]:
                    if k in (i, h):
                        continue
                    di_key = (h, i, j, k)
                    rev_key = (k, j, i, h)
                    if di_key not in seen_dihedrals and rev_key not in seen_dihedrals:
                        val = _measure_dihedral(coords, h, i, j, k)
                        label = f"{elems[h]}-{elems[i]}-{elems[j]}-{elems[k]}"
                        logger.info(f"{'Dihedral':<10} {di_key!s:<20} {label:<15} {val:>10.2f}°")
                        seen_dihedrals.add(di_key)

    logger.info("%s", "-" * 60)


def print_internal_coordinate_table(fragment, actatoms=None) -> None:
    """Log a table of bonds, angles and dihedrals for a fragment."""
    timeA = time.time()
    logger.info("\nPrinting internal coordinate table")
    if actatoms is not None:
        logger.info("Actatoms: %s", actatoms)

    if actatoms is None:
        actatoms = []
        chosen_coords = fragment.coords
        chosen_elems = fragment.elems

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

    connectivity = _calc_conn_py(chosen_coords, chosen_elems, conndepth, scale, tol)
    logger.info("Connectivity calculation complete.")

    bondpairsdict = {}

    for conn_fragment in connectivity:
        for atom in conn_fragment:
            connatoms = get_connected_atoms(chosen_coords, chosen_elems, CONNECTIVITY_SCALE, CONNECTIVITY_TOL, atom)
            for conn_i in connatoms:
                dist = distance(chosen_coords[atom], chosen_coords[conn_i])
                bondpairsdict[frozenset((atom, conn_i))] = dist

    logger.info(small_header("Internal coordinates"))

    logger.info(small_header("Bond lengths (Å):"))
    for key, val in bondpairsdict.items():
        listkey = list(key)
        elA = chosen_elems[listkey[0]]
        elB = chosen_elems[listkey[1]]
        if not actatoms:
            logger.info(f"Bond: {listkey[0]:8}{elA:4} - {listkey[1]:4}{elB:4} {val:>6.3f}")
        else:
            fullsystem_keyA = actatoms[listkey[0]]
            fullsystem_keyB = actatoms[listkey[1]]
            if fullsystem_keyA in actatoms and fullsystem_keyB in actatoms:
                logger.info(f"Bond: {fullsystem_keyA:8}{elA:4} - {fullsystem_keyB:4}{elB:4} {val:>6.3f}")
    logger.info("%s", "=" * 50)
    log_time_since(timeA, "print internal coordinate table")


def print_coords_for_atoms(coords, elems, members, labels=None):
    if labels is not None and len(labels) != len(members):
        raise InputError("Problem. Length of Labels note equal to length of members list")
    label = ""
    for i, m in enumerate(members):
        if labels is not None:
            label = labels[i]
        logger.info(f"{label:>4} {elems[m]:>4} {coords[m][0]:>12.8f}  {coords[m][1]:>12.8f}  {coords[m][2]:>12.8f}")


def write_xyz_for_atoms(coords, elems, members, name):
    subset_elems = [elems[i] for i in members]
    subset_coords = np.take(coords, members, axis=0)
    with open(name + ".xyz", "w") as ofile:
        ofile.write(str(len(subset_elems)) + "\n")
        ofile.write("title" + "\n")
        for el, c in zip(subset_elems, subset_coords, strict=False):
            line = f"{el:4} {c[0]:>12.6f} {c[1]:>12.6f} {c[2]:>12.6f}"
            ofile.write(line + "\n")


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


def distance(A, B):
    return sqrt((A[0] - B[0]) ** 2 + (A[1] - B[1]) ** 2 + (A[2] - B[2]) ** 2)  # fastest


def angle(A, B, C):
    AB = A - B
    CB = C - B
    dot_product = np.dot(AB, CB)
    magnitude1 = np.linalg.norm(AB)
    magnitude2 = np.linalg.norm(CB)
    angle_rad = np.arccos(dot_product / (magnitude1 * magnitude2))
    return np.degrees(angle_rad)


def dihedral(A, B, C, D):
    v1 = B - A
    v2 = C - B
    v3 = D - C

    n1 = np.cross(v1, v2)
    n2 = np.cross(v2, v3)

    dot = np.dot(n1, n2)
    if dot < 0:
        dihedral_angle = -1 * (np.arccos(dot / (np.linalg.norm(n1) * np.linalg.norm(n2))))
    else:
        dihedral_angle = np.arccos(dot / (np.linalg.norm(n1) * np.linalg.norm(n2)))

    return dihedral_angle * 180 / np.pi


def distance_between_atoms(fragment=None, atoms=None) -> float:
    """Return the distance between two atoms of a fragment."""
    return distance(fragment.coords[atoms[0]], fragment.coords[atoms[1]])


def angle_between_atoms(fragment=None, atoms=None) -> float:
    """Return the A-B-C angle spanned by three atoms of a fragment."""
    return angle(fragment.coords[atoms[0]], fragment.coords[atoms[1]], fragment.coords[atoms[2]])


def dihedral_between_atoms(fragment=None, atoms=None) -> float:
    """Return the A-B-C-D dihedral angle spanned by four atoms of a fragment."""
    return dihedral(
        fragment.coords[atoms[0]], fragment.coords[atoms[1]], fragment.coords[atoms[2]], fragment.coords[atoms[3]]
    )


def get_centroid(coords):
    sum_x = 0
    sum_y = 0
    sum_z = 0
    for c in coords:
        sum_x += c[0]
        sum_y += c[1]
        sum_z += c[2]
    return [sum_x / len(coords), sum_y / len(coords), sum_z / len(coords)]


def threshold_conn(elA, elB, scale, tol):
    return scale * (eldict_covrad[elA] + eldict_covrad[elB]) + tol


def _calc_conn_py(coords, elems, conndepth, scale, tol):
    found_atoms = []
    fraglist = []
    for atom in range(len(elems)):
        if atom not in found_atoms:
            members = _get_molecule_members_np(coords, elems, conndepth, scale, tol, atomindex=atom)
            if members not in fraglist:
                fraglist.append(members)
                found_atoms += members
    return fraglist


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


# https://semantive.com/pl/blog/high-performance-computation-in-python-numpy/
def _einsum_mat(mat_v, mat_u):
    mat_z = mat_v - mat_u
    return np.sqrt(np.einsum("ij,ij->i", mat_z, mat_z))


# https://semantive.com/pl/blog/high-performance-computation-in-python-numpy/
def _get_connected_atoms_np(coords, elems, scale, tol, atomindex):
    compcoords = np.tile(coords[atomindex], (len(coords), 1))
    distances = _einsum_mat(coords, compcoords)
    el_covrad_ref = eldict_covrad[elems[atomindex]]
    # Cheaper way of getting thresholds list than calling threshold_conn
    thresholds = np.array([eldict_covrad[elems[i]] for i in range(len(elems))])
    thresholds = thresholds + el_covrad_ref
    thresholds = thresholds * scale
    thresholds = thresholds + tol
    diff = distances - thresholds
    return np.where(diff < 0)[0].tolist()


def get_connected_atoms_dict(coords, elems, scale, tol):
    conndict = {}
    for c in range(len(coords)):
        conn = _get_connected_atoms_np(coords, elems, scale, tol, c)
        conn.remove(c)
        conndict[c] = conn
    return conndict


# Version 2 never goes through same atom


def _get_molecule_members_np(coords, elems, loopnumber, scale, tol, atomindex=None, membs=None):
    if membs is None:
        membs = []
        membs.append(atomindex)
        membs = _get_connected_atoms_np(coords, elems, scale, tol, atomindex)

    if isinstance(membs, int):
        membs = [membs]
    finalmembs = membs

    for _i in range(loopnumber):
        newmembers = [_get_connected_atoms_np(coords, elems, scale, tol, k) for k in membs]
        trimmed_flat = np.unique([item for sublist in newmembers for item in sublist]).tolist()

        membs = listdiff(trimmed_flat, finalmembs)
        if len(membs) == 0:
            return finalmembs
        finalmembs += membs
        finalmembs = np.unique(finalmembs).tolist()
    return finalmembs


def elems_to_formula(elems):
    # Counting once per unique element rather than per atom: elems can be very long
    counts = Counter(elems)
    ordered = []
    if "C" in counts:
        ordered.append("C")
        if "H" in counts:
            ordered.append("H")
    ordered += sorted(element for element in counts if element not in ordered)
    return "".join(f"{element}{counts[element]}" for element in ordered)


def _formula_to_elem_list(formulastring):
    el = ""
    diff = ""
    els = []
    atomunits = []
    numels = []
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
    els.reverse()
    numels.reverse()
    atoms.reverse()
    return atoms


def read_xyzfile(filename) -> tuple[list[str], np.ndarray]:
    """Read elements and coordinates from an XYZ file."""
    logger.info(f"Reading coordinates from XYZ file '{filename}'.")
    coords = []
    elems = []
    with open(filename) as f:
        for count, line in enumerate(f):
            if count == 0:
                numatoms = int(line.split()[0])
            if count > 1 and len(line.strip()) > 0:
                if isint(line.split()[0]) is True:
                    el = reformat_element(int(line.split()[0]), isatomnum=True)
                    elems.append(el)
                else:
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


def read_xyzfiles(xyzdir, readchargemult=False) -> list:
    """Create a Fragment for every XYZ file in a directory."""
    logger.info("read_xyzfiles function")
    logger.info("Note: will read XYZ-files in directory using natural sorting")
    import glob

    filenames = []
    fragments = []
    for file in natural_sort(glob.glob(xyzdir + "/*.xyz")):
        filename = os.path.basename(file)
        filenames.append(filename)
        logger.info("\n\nXYZ-file: %s", filename)
        mol = openmmqmmm.Fragment(xyzfile=file, readchargemult=readchargemult, label=filename)
        fragments.append(mol)
    return fragments


def write_xyzfile(elems, coords, name, writemode="w", title="title") -> None:
    """Write elements and coordinates to an XYZ file."""
    header = [f"{len(elems)}\n", f"{title}\n"]
    atomlines = [f"{el:4} {c[0]:16.12f} {c[1]:16.12f} {c[2]:16.12f}\n" for el, c in zip(elems, coords, strict=False)]
    with open(name + ".xyz", writemode) as ofile:
        ofile.writelines(header)
        ofile.writelines(atomlines)
    logger.info("Wrote XYZ file:  %s", name + ".xyz")


# Also grabs last word in title line. Typically an energy (has to be converted to float outside)
def split_multimolxyzfile(file, writexyz=False, skipindex=1, return_fragments=False) -> list | tuple[list, list, list]:
    """Split a multi-molecule XYZ file (trajectory, conformer set) into its frames."""
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
                        write_xyzfile(elems, coords, "molecule" + str(molcounter))
                    frag = Fragment(coords=coords, elems=elems)
                    fragments.append(frag)
                    coords = []
                    elems = []
            if titlegrab is True:
                if len(line.split()) > 0:
                    all_titles.append(line.split())
                else:
                    all_titles.append("NA")
                titlegrab = False
                coordgrab = True
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


def _read_chemshellfragfile_xyz(fragfile):
    pathtofragfile = fragfile.split(".")[0] + ".c"
    coords = []
    elems = []
    grabcoords = False
    with open(pathtofragfile) as ffile:
        for line in ffile:
            if "block = connectivity" in line:
                grabcoords = False
            if grabcoords is True:
                coords.append([float(i) * openmmqmmm.constants.BOHR_TO_ANG for i in line.split()[1:]])
                el = reformat_element(line.split()[0])
                elems.append(el)
            if "block = coordinates records " in line:
                grabcoords = True
        coords = _reformat_list_to_array(coords)
    return elems, coords


def _conv_atomtypes_elems(atomtype):
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


def read_gromacsfile(grofile) -> tuple[list[str], np.ndarray, list]:
    """Read a GROMACS .gro coordinate file."""
    elems = []
    coords = []
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
                box_dims = [10 * float(i) for i in line.split()]
                # Assuming cubic and adding 90,90,90
                box_dims.append(90.0)
                box_dims.append(90.0)
                box_dims.append(90.0)
                logger.info("Box dimensions read:  %s", box_dims)
            else:
                linelist = line.split()
                atomtype = linelist[1]
                atomtype = "".join(item for item in atomtype if not item.isdigit())
                atomtype = atomtype.replace("'", "")
                elem = _conv_atomtypes_elems(atomtype)
                elems.append(elem)

                # If larer than 7 then GRO file contains both coords and velocities
                if len(linelist) > 7:
                    coords_x = float(linelist[-6])
                    coords_y = float(linelist[-5])
                    coords_z = float(linelist[-4])
                else:
                    coords_x = float(linelist[-3])
                    coords_y = float(linelist[-2])
                    coords_z = float(linelist[-1])
                # Converting from nm to Ang
                coords.append([10 * coords_x, 10 * coords_y, 10 * coords_z])
    npcoords = _reformat_list_to_array(coords)
    if len(npcoords) != len(elems):
        raise FileFormatError(f"Num coords not equal to num elems. Parsing of Gromacsfile: {grofile} failed. BUG!")
    return elems, npcoords, box_dims


def read_ambercoordinates(prmtopfile=None, inpcrdfile=None) -> tuple[list[str], np.ndarray, list]:
    """Read an Amber inpcrd/rst coordinate file, taking elements from the prmtop."""
    elems = []
    coords = []
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


def write_pdbfile(
    fragment,
    *,
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
    """Write a fragment to a PDB file."""
    logger.info("Writing PDB-file...")
    elems = fragment.elems
    coords = fragment.coords

    # NOTE: These lists are only defined for CHARMM files currently. Not Amber or GROMACS
    if openmmobject is not None:
        atomnames = openmmobject.atomnames
        resnames = openmmobject.resnames
        residlabels = openmmobject.resids
        segmentlabels = openmmobject.segmentnames

    if atomnames is None or len(atomnames) == 0:
        logger.warning("Using elements as atomnames")
        atomnames = fragment.elems
    if resnames is None or len(resnames) == 0:
        resnames = fragment.numatoms * [dummyname]
    if chainlabels is None or len(chainlabels) == 0:
        chainlabels = fragment.numatoms * [""]
    if residlabels is None or len(residlabels) == 0:
        residlabels = fragment.numatoms * [1]
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

            resname_short = resname[0:3]

            atomnamestring = atomname[-4:]

            if not any(char.isdigit() for char in atomnamestring):
                atomnamestring = atomnamestring + str(count + 1)

            # Using string format from: cupnet.net/pdb-format/

            resid_str = str(resid)

            if charges_column is not None:
                charge = charges_column[count]
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
            else:
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
        if conect_lines is not None:
            pfile.writelines(conect_lines)
    logger.info("Wrote PDB file:  %s", outputname + ".pdb")
    return outputname + ".pdb"


def total_nuclear_charge(ellist):
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


def elems_to_nuclear_charges(ellist):
    nuccharges = []
    for e in ellist:
        atcharge = elematomnumbers[e.lower()]
        nuccharges.append(atcharge)
    return nuccharges


def total_mass(ellist):
    return sum(list_of_masses(ellist))


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


def flexible_align_xyz(
    xyzfile_a, xyzfile_b, rotate_only=False, translate_only=False, reordering=False, reorder_method="brute", subset=None
) -> None:
    """Align the molecule in one XYZ file onto the molecule in another."""
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

    newfragA.write_xyzfile(f"{xyzfile_a.replace('.xyz', '')}_aligned.xyz")


def flexible_align_pdb(
    pdbfileA, pdbfileB, rotate_only=False, translate_only=False, reordering=False, reorder_method="brute", subset=None
) -> None:
    """Align the molecule in one PDB file onto the molecule in another."""
    logger.info(f"Will align molecule in file {pdbfileA} onto molecule in file {pdbfileB}")
    fragment_a = Fragment(pdbfile=pdbfileA)
    fragment_b = Fragment(pdbfile=pdbfileB)

    newfragA = flexible_align(
        fragment_a,
        fragment_b,
        rotate_only=rotate_only,
        translate_only=translate_only,
        reordering=reordering,
        reorder_method=reorder_method,
        subset=subset,
    )

    fragment_a.coords = newfragA.coords  # Replacing coords in original fragmentA
    fragment_a.write_pdbfile_openmm(filename=f"{pdbfileA.replace('.pdb', '')}_aligned")  # Now write out


def _resolve_alignment_subsets(fragment_a, fragment_b, subset, heavyatomsonly=False):
    """Return the (coords, elems) of each fragment that a superposition should be fitted on."""
    if subset is None:
        if heavyatomsonly is not True:
            return fragment_a.coords, fragment_a.elems, fragment_b.coords, fragment_b.elems
        indices_a = fragment_a.get_non_h_atomindices()
        indices_b = fragment_b.get_non_h_atomindices()
        coords_a, elems_a = fragment_a.get_coords_for_atoms(indices_a)
        coords_b, elems_b = fragment_b.get_coords_for_atoms(indices_b)
        return coords_a, elems_a, coords_b, elems_b

    if any(isinstance(el, list) for el in subset):
        subset_a, subset_b = subset[0], subset[1]
        if len(subset_a) != len(subset_b):
            raise InputError("Length of subsets not equal. This is not allowed. Exiting.")
    else:
        # One index list for both fragments; only correct when their atom order matches
        subset_a = subset_b = subset

    coords_a, elems_a = fragment_a.get_coords_for_atoms(subset_a)
    coords_b, elems_b = fragment_b.get_coords_for_atoms(subset_b)
    logger.debug("Alignment subset A: %s %s", elems_a, coords_a)
    logger.debug("Alignment subset B: %s %s", elems_b, coords_b)
    return coords_a, elems_a, coords_b, elems_b


def flexible_align(
    fragment_a,
    fragment_b,
    rotate_only=False,
    translate_only=False,
    reordering=False,
    reorder_method="brute",
    subset=None,
) -> "Fragment":
    """Align one fragment onto another (Kabsch superposition, optional reordering)."""
    logger.info("flexible_align function")
    import geometric

    subsetA_coords, subsetA_elems, subsetB_coords, subsetB_elems = _resolve_alignment_subsets(
        fragment_a, fragment_b, subset
    )

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
        order = _reorder(
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

    trans, rot = geometric.molecule.get_rotate_translate(subsetA_coords, subsetB_coords)

    if translate_only is True:
        logger.info("Doing translation only")
        Anew = fragment_a.coords + trans
    elif rotate_only is True:
        logger.info("Doing rotation only")
        Anew = np.dot(fragment_a.coords, rot)
    else:
        Anew = np.dot(fragment_a.coords, rot) + trans

    newfrag = Fragment(elems=fragment_a.elems, coords=Anew)
    logger.info("New aligned structure")
    newfrag.print_coords()

    return newfrag


# NOTE: no reordering
def calculate_rmsd(fragment_a, fragment_b, subset=None, heavyatomsonly=False, write_aligned_structure=False) -> float:
    """Return the RMSD between two fragments after optimal superposition."""
    logger.info("calculate_RMSD function")

    subsetA_coords, _elems_a, subsetB_coords, _elems_b = _resolve_alignment_subsets(
        fragment_a, fragment_b, subset, heavyatomsonly=heavyatomsonly
    )

    import geometric

    trans, rot = geometric.molecule.get_rotate_translate(subsetA_coords, subsetB_coords)
    Anew = np.dot(subsetA_coords, rot) + trans

    rmsdval = float(np.sqrt(((Anew - subsetB_coords) ** 2).sum() / len(Anew)))

    logger.info("RMSD: %s", rmsdval)

    if write_aligned_structure:
        logger.info("write_aligned_structure active")
        newfrag = Fragment(elems=fragment_a.elems, coords=Anew)
        newfrag.write_xyzfile("structA_aligned.xyz")

    return rmsdval


def _centroid(X):
    return X.mean(axis=0)


def get_partial_list(allatoms, partialatoms, full_list):
    newlist = copy.copy(full_list)  # Otherwise object may be updated
    otheratoms = listdiff(allatoms, partialatoms)
    otheratoms.reverse()
    for at in otheratoms:
        del newlist[at]
    return newlist


def _reorder(reorder_method, p_coord, q_coord, p_atoms, q_atoms):
    p_cent = _centroid(p_coord)
    q_cent = _centroid(q_coord)
    p_coord -= p_cent
    q_coord -= q_cent

    p_atoms = np.array([elematomnumbers[el.lower()] for el in p_atoms])
    q_atoms = np.array([elematomnumbers[el.lower()] for el in q_atoms])

    q_review = reorder_method(p_atoms, q_atoms, p_coord, q_coord)
    return [q_review.tolist()][0]


def expand_qm_region(fragment=None, initial_atoms=None, radius=None) -> list[int]:
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
                    if len(fragment.connectivity) == 0:
                        wholemol = _get_molecule_members_np(
                            fragment.coords, fragment.elems, 99, scale, tol, atomindex=index
                        )
                    else:
                        for q in fragment.connectivity:
                            if index in q:
                                wholemol = q
                                break

                    atomlist = atomlist + wholemol
    return np.unique(atomlist).tolist()


def expand_qm_pc_region(theory=None, fragment=None, thresh=5e-4) -> list[int]:
    """Expand a QM region based on the QM/MM pointcharge-gradient magnitude."""
    if theory is None and fragment is None:
        raise InputError("QMPC_fragexpand requires fragment and theory")
    if not isinstance(theory, openmmqmmm.QMMMTheory):
        raise InputError("Theory is not a QMMMTheory")

    openmmqmmm.single_point(theory=theory, fragment=fragment, grad=True)

    pcgrad = theory.PCgradient
    large_force_indices = np.unique(np.argwhere(abs(pcgrad) > thresh)[:, 0])
    proper_largeforce_indices = large_force_indices + len(theory.qmatoms)
    fragment.calc_connectivity()  # get connectivity

    new_expansion = theory.qmatoms

    for i in proper_largeforce_indices:
        mol_index = search_list_of_lists_for_index(i, fragment.connectivity)
        molmembers = fragment.connectivity[mol_index]
        new_expansion = new_expansion + molmembers
    new_expansion = np.unique(new_expansion)

    logger.info("New QM-region expansion based on pointcharge gradient selection")
    fragment.print_coords_for_atoms(new_expansion, labels=new_expansion)
    logger.info("Writing coordinates to file: QMPC_selection.xyz")
    fragment.write_xyz_for_atoms(xyzfilename="QMPC_selection.xyz", atoms=new_expansion)

    return new_expansion


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
        connatoms = _get_connected_atoms_np(coords, elems, scale, tol, qmatom)
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
            qm_mm_boundary_dict[qmatom] = boundaryatom
        elif len(boundaryatom) == 1:
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
            qm_mm_boundary_dict[qmatom] = [boundaryatom[0]]
    logger.info("QM-MM boundary dictionary: %s", qm_mm_boundary_dict)
    log_time_since(timeA, "get_boundary_atoms")
    return qm_mm_boundary_dict


# Two methods: simple method (default) and ratio method.
# Simple method: Just use a fixed distance (default 1.09 Å)
# Ratio method: Determine by scaling QM1-MM1 distance with a ratio. Ratio can be fixed value (e.g. 0.723) or determined
# from equilibrium distances (not ready)
# Using linkatom distance of 1.09 Å for now as default. Makes sense for C-H link atoms.
def get_linkatom_positions(
    qm_mm_boundary_dict,
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
    linkdistances_dict = {("C", "H"): 1.09, ("O", "H"): 0.98, ("N", "H"): 0.99}
    logger.info("Linkatom distance dictionary: %s", linkdistances_dict)
    if linkatom_method == "ratio" and linkatom_ratio == "Auto" and bondpairs_eq_dict is None:
        bondpairs_eq_dict = {
            ("C", "H"): 1.09,
            ("C", "C"): 1.522269,
            ("C", "N"): 1.47,
            ("C", "O"): 1.43,
            ("C", "S"): 1.81,
        }

    logger.info("qm_mm_boundary_dict: %s", qm_mm_boundary_dict)
    linkatoms_dict = {}
    for dict_item in qm_mm_boundary_dict.items():
        qmatom = dict_item[0]
        for mmatom in dict_item[1]:
            qmatom_coords = np.array(coords[qmatom])
            mmatom_coords = np.array(coords[mmatom])
            if linkatom_method == "ratio":
                if linkatom_ratio == "Auto":
                    logger.info("Automatic ratio. Determining ratio based on dict of equilibrium distances")
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
                    linkatom_distance = linkdistances_dict[(elems[qmatom], linkatom_type)]
                else:
                    linkatom_distance = linkatom_simple_distance
                logger.info("Linkatom distance (QM1-L) is: %s", linkatom_distance)
                linkatom_coords = list(
                    qmatom_coords
                    + (mmatom_coords - qmatom_coords) * (linkatom_distance / distance(qmatom_coords, mmatom_coords))
                )
            else:
                raise InputError("Invalid linkatom_method. Exiting.")

            linkatoms_dict[(qmatom, mmatom)] = linkatom_coords
    return linkatoms_dict


def get_molecules_from_trajectory(file, writexyz=False, skipindex=1, conncalc=False) -> list:
    """Create a Fragment for every snapshot in a multi-molecule XYZ trajectory."""
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


def get_water_constraints(openmmtheoryobject=None, atomlist=None, watermodel="tip3p") -> list:
    """Return bond constraints for every water molecule in an OpenMM system."""
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
            if index not in atomlist:
                continue

            if rn in water_resname and el == "O":
                waterconstraints.append([index, index + 1])
                waterconstraints.append([index, index + 2])
                waterconstraints.append([index + 1, index + 2])

    return waterconstraints


def _qm_region_owner(theory, max_depth=4):
    # A job may be handed a wrapper (NumGrad) whose theorytype is that of no particular theory,
    # so the object owning the QM-region charge is found by capability, not by that string.
    for _ in range(max_depth):
        if theory is None:
            return None
        if hasattr(theory, "resolve_qm_charge_mult"):
            return theory
        theory = getattr(theory, "theory", None)
    return None


def check_charge_mult(charge, mult, theorytype, fragment, jobtype, theory=None):
    """Resolve the charge and multiplicity a theory should be run with."""
    qm_region = _qm_region_owner(theory)
    if qm_region is not None:
        return qm_region.resolve_qm_charge_mult(charge=charge, mult=mult)

    if theorytype == "QM":
        # A plain QM theory treats the whole fragment as its system, so the fragment's own
        # charge/mult are the right fallback here.
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
        raise InternalError(
            f"{jobtype} was given a QM/MM theory that cannot resolve its QM-region charge. A QM/MM theory must "
            f"provide a resolve_qm_charge_mult method."
        )
    elif theorytype == "MM":
        charge = None
        mult = None
    return charge, mult


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


def define_xh_constraints(fragment, actatoms=None, excludeatoms=None) -> list:
    """Return X-H bond constraints for a fragment or a region of it."""
    logger.info("Inside define_XH_constraints function")
    if actatoms is None:
        subset_elems = fragment.elems
        subset_coords = fragment.coords
        actatoms = fragment.atomlist
    else:
        subset_elems = [fragment.elems[i] for i in actatoms]
        subset_coords = np.take(fragment.coords, actatoms, axis=0)

    logger.info(f"Defining constraints for {len(subset_elems)} atom-region")

    tempHatoms = [index for index, el in enumerate(subset_elems) if el == "H"]
    tempHatoms_full = [_actindex_to_fullindex(i, actatoms) for i in tempHatoms]
    Hatoms = []
    if excludeatoms is not None:
        logger.info("Checking for exclude atoms")
        for th, th_f in zip(tempHatoms, tempHatoms_full, strict=False):
            if th_f not in excludeatoms:
                Hatoms.append(th)
    else:
        Hatoms = tempHatoms

    # py version (slow) but good enough for a few thousand atoms
    scale = CONNECTIVITY_SCALE
    tol = CONNECTIVITY_TOL
    act_con_list = []
    for Hatom in Hatoms:
        connatoms = _get_connected_atoms_np(subset_coords, subset_elems, scale, tol, Hatom)
        act_con_list.append(connatoms)
    final_list = []
    for XHpair in act_con_list:
        if len(XHpair) != 2:
            raise InternalError(f"XHpair is strange: {XHpair}")
        final_list.append([_actindex_to_fullindex(XHpair[0], actatoms), _actindex_to_fullindex(XHpair[1], actatoms)])
    return final_list


def fullindex_to_actindex(fullindex, actatoms):
    return actatoms.index(fullindex)


def _actindex_to_fullindex(actindex, actatoms):
    return actatoms[actindex]


# Limitation: Assumes all waters from starting index to end and that waters are ordered: O H H
def simple_get_water_constraints(fragment, starting_index=None, onlyHH=False) -> list:
    """Return water bond constraints by position, without residue information."""
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
            if onlyHH is False:
                constraints.append([i, i + 1])
                constraints.append([i, i + 2])
            # H-H constraints. i.e. effectively freezing angles
            constraints.append([i + 1, i + 2])
    return constraints


def _combine_and_place_fragments(ref_frag, trans_frag):
    for displacement in [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0]:
        trans_frag.coords[:, -1] += displacement
        members = _get_molecule_members_np(
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


# Use tolerance (tol) e.g. to control how many solvent molecules around get deleted
# Currently using 0.4 as default based on threonine in acetonitrile example
def insert_solute_into_solvent(
    *,
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
    """Insert one or two solute molecules into a solvent box, removing clashes."""
    logger.info("\ninsert_solute_into_solvent\n")
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

    com_box = solvent.get_coordinate_center()

    if solute2 is None:
        com_solute = solute.get_coordinate_center()
        trans_coord = np.array(com_box) - np.array(com_solute)
        solute.coords = solute.coords + trans_coord
    else:
        combined_solute = _combine_and_place_fragments(ref_frag=solute, trans_frag=solute2)

        com_solute = combined_solute.get_coordinate_center()
        trans_coord = np.array(com_box) - np.array(com_solute)
        combined_solute.coords = combined_solute.coords + trans_coord

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

    membs = _get_molecule_members_np(new_frag.coords, new_frag.elems, 20, scale, tol, atomindex=0, membs=None)
    delatoms = [i for i in membs if i >= solute.numatoms]
    logger.info("First delatoms: %s", delatoms)
    if solute2 is not None:
        membs2 = _get_molecule_members_np(
            new_frag.coords, new_frag.elems, 20, scale, tol, atomindex=solute.numatoms, membs=None
        )
        logger.info("membs2: %s", membs2)
        for j in membs2:
            if j >= solute.numatoms + solute2.numatoms and j not in delatoms:
                delatoms.append(j)
    logger.info("Final delatoms: %s", delatoms)

    delatoms.sort(reverse=True)
    logger.info("Found clashing solvent atoms: %s", delatoms)
    for d in delatoms:
        new_frag.delete_atom(d)
    logger.info("")
    logger.info("Final fragment after removing clashing atoms:")
    new_frag.update_attributes()
    new_frag.write_xyzfile(xyzfilename="solution.xyz")

    if write_pdb:
        logger.info("Write_PDB is active. Will write PDB-file of solute+solvent system for topology purposes")
        try:
            import openmm.app
        except ImportError:
            raise MissingDependencyError("Error: OpenMM library not found. Please install OpenMM") from None

        pdb1 = openmm.app.PDBFile(solute_pdb)
        solute_resname = next(iter(pdb1.topology.residues())).name
        logger.info("solute_resname: %s", solute_resname)
        pdb2 = openmm.app.PDBFile(solvent_pdb)
        solvent_box_vectors = pdb2.topology.getPeriodicBoxVectors()
        logger.info("Found PBC vectors in solvent PDB-file: %s", solvent_box_vectors)

        modeller = openmm.app.Modeller(pdb1.topology, pdb1.positions)  # Add pdbfile1

        if solute2 is not None:
            logger.info("Adding solute2")
            pdb_solute2 = openmm.app.PDBFile(solute2_pdb)
            solute2_resname = next(iter(pdb_solute2.topology.residues())).name
            logger.info("solute2_resname: %s", solute2_resname)
            modeller.add(pdb_solute2.topology, pdb_solute2.positions)  # Add pdbfile2
        logger.info("Adding solvent")
        modeller.add(pdb2.topology, pdb2.positions)  # Add pdbfile2

        toDelete = [r for j, r in enumerate(modeller.topology.atoms()) if j in delatoms]
        modeller.delete(toDelete)
        mergedPositions = new_frag.coords

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

        if write_pbc_info:
            logger.info("write_PBC_info True: Writing PBC to header of PDB-file")
            if solvent_box_vectors is not None:
                logger.info("PBC vectors found in solvent PDB-file: %s", solvent_box_vectors)
                logger.info("Adding to solution PDB-file")
                modeller.topology.setPeriodicBoxVectors(solvent_box_vectors)

        openmmqmmm.openmm.write_pdbfile_openmm_topology(modeller.topology, mergedPositions, outputname)
    return new_frag


# Assumes coords in Angstrom
def nuc_nuc_repulsion(coords, charges) -> float:
    """Return the classical nucleus-nucleus repulsion energy of a set of point charges."""
    charges = np.array(charges)  # Ensure charges is a numpy array
    coords_b = coords * openmmqmmm.constants.ANG_TO_BOHR
    diff = coords_b[:, None, :] - coords_b[None, :, :]
    distances = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(distances, np.inf)
    return 0.5 * np.sum(charges[:, None] * charges[None, :] / distances)


def define_dummy_topology(elems, resname="MOL"):
    try:
        import openmm.app
    except ImportError:
        raise InputError("Error: OpenMM not found. Cannot define a topology") from None
    logger.info("Defining new basic single-chain, multi-residue topology")
    pdb_topology = openmm.app.Topology()
    chain = pdb_topology.addChain()
    residue = pdb_topology.addResidue(resname, chain)

    atomnames_dict = defaultdict(int)
    for el in elems:
        atomnumber = openmm.app.Element.getBySymbol(el).atomic_number
        element = openmm.app.Element.getByAtomicNumber(atomnumber)
        atomnames_dict[el] += 1
        atomname = f"{el}{atomnames_dict[el]}"
        pdb_topology.addAtom(atomname, element, residue)
    return pdb_topology

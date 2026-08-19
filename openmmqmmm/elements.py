from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Element:
    """One periodic-table entry: name, symbol and atomic number."""

    name: str
    symbol: str
    atomnumber: int


# fmt: off
_ELEMENT_TABLE = (
    (  0, "M",  "dummy",            0.0,        None, None,    None),
    (  1, "H",  "hydrogen",        0.31,     1.00794, 0.32,  0.0056),
    (  2, "He", "helium",          0.28,    4.002602, 0.37, -0.1543),
    (  3, "Li", "lithium",         1.28,        6.94,  1.3,     0.0),
    (  4, "Be", "beryllium",       0.96,   9.0121831, 0.99,  0.0333),
    (  5, "B",  "boron",           0.84,       10.81, 0.84,  -0.103),
    (  6, "C",  "carbon",          0.76,     12.0107, 0.75, -0.0446),
    (  7, "N",  "nitrogen",        0.71,     14.0067, 0.71, -0.1072),
    (  8, "O",  "oxygen",          0.66,     15.9994, 0.64, -0.0802),
    (  9, "F",  "fluorine",        0.57, 18.99840316,  0.6, -0.0629),
    ( 10, "Ne", "neon",            0.58,     20.1797, 0.62, -0.1088),
    ( 11, "Na", "sodium",        0.0001, 22.98976928,  1.6,  0.0184),
    ( 12, "Mg", "magnesium",       1.41,      24.305,  1.4,     0.0),
    ( 13, "Al", "aluminum",        1.21,  26.9815385, 1.24, -0.0726),
    ( 14, "Si", "silicon",         1.11,      28.085, 1.14,  -0.079),
    ( 15, "P",  "phosphorus",      1.07,   30.973762, 1.09, -0.0756),
    ( 16, "S",  "sulfur",          1.05,      32.065, 1.04, -0.0565),
    ( 17, "Cl", "chlorine",        1.02,       35.45,  1.0, -0.0444),
    ( 18, "Ar", "argon",           1.06,      39.948, 1.01, -0.0767),
    ( 19, "K",  "potassium",     0.0001,     39.0983,  2.0,   0.013),
    ( 20, "Ca", "calcium",         1.76,      40.078, 1.74,     0.0),
    ( 21, "Sc", "scandium",         1.7,   44.955908, 1.59,     0.0),
    ( 22, "Ti", "titanium",         1.6,      47.867, 1.48,     0.0),
    ( 23, "V",  "vanadium",        1.53,     50.9415, 1.44,     0.0),
    ( 24, "Cr", "chromium",        1.39,     51.9961,  1.3,     0.0),
    ( 25, "Mn", "manganese",       1.61,   54.938044, 1.29,     0.0),
    ( 26, "Fe", "iron",            1.52,      55.845, 1.24,     0.0),
    ( 27, "Co", "cobalt",           1.5,   58.933194, 1.18,     0.0),
    ( 28, "Ni", "nickel",          1.24,     58.6934, 1.17,     0.0),
    ( 29, "Cu", "copper",          1.32,      63.546, 1.22,     0.0),
    ( 30, "Zn", "zinc",            1.22,       65.38,  1.2,     0.0),
    ( 31, "Ga", "gallium",         1.22,      69.723, 1.23, -0.0512),
    ( 32, "Ge", "germanium",        1.2,       72.63,  1.2, -0.0557),
    ( 33, "As", "arsenic",         1.19,   74.921595,  1.2, -0.0533),
    ( 34, "Se", "selenium",         1.2,      78.971, 1.18, -0.0399),
    ( 35, "Br", "bromine",          1.2,      79.904, 1.17, -0.0313),
    ( 36, "Kr", "krypton",         1.16,      83.798, 1.16, -0.0541),
    ( 37, "Rb", "rubidium",         2.2,     85.4678, 2.15,  0.0092),
    ( 38, "Sr", "strontium",       1.95,       87.62,  1.9,     0.0),
    ( 39, "Y",  "yttrium",          1.9,    88.90584, 1.76,     0.0),
    ( 40, "Zr", "zirconium",       1.75,      91.224, 1.64,     0.0),
    ( 41, "Nb", "niobium",         1.64,    92.90637, 1.56,     0.0),
    ( 42, "Mo", "molybdenum",      1.54,       95.96, 1.46,     0.0),
    ( 43, "Tc", "technetium",      1.47,          97, 1.38,     0.0),
    ( 44, "Ru", "ruthenium",       1.46,      101.07, 1.36,     0.0),
    ( 45, "Rh", "rhodium",         1.42,    102.9055, 1.34,     0.0),
    ( 46, "Pd", "palladium",       1.39,      106.42,  1.3,     0.0),
    ( 47, "Ag", "silver",          1.45,    107.8682, 1.36,     0.0),
    ( 48, "Cd", "cadmium",         1.44,     112.414,  1.4,     0.0),
    ( 49, "In", "indium",          1.42,     114.818, 1.42, -0.0361),
    ( 50, "Sn", "tin",             1.39,      118.71,  1.4, -0.0393),
    ( 51, "Sb", "antimony",        1.39,      121.76,  1.4, -0.0376),
    ( 52, "Te", "tellurium",       1.38,       127.6, 1.37, -0.0281),
    ( 53, "I",  "iodine",          1.39,   126.90447, 1.36,  -0.022),
    ( 54, "Xe", "xenon",            1.4,     131.293, 1.36, -0.0381),
    ( 55, "Cs", "cesium",          2.44,  132.905452, 2.38,  0.0065),
    ( 56, "Ba", "barium",          2.15,     137.327, 2.06,     0.0),
    ( 57, "La", "lanthanum",       2.07,   138.90547, 1.94,     0.0),
    ( 58, "Ce", "cerium",          2.04,     140.116, 1.84,     0.0),
    ( 59, "Pr", "praseodymium",    2.03,   140.90766,  1.9,     0.0),
    ( 60, "Nd", "neodymium",       2.01,     144.242, 1.88,     0.0),
    ( 61, "Pm", "promethium",      1.99,         145, 1.86,     0.0),
    ( 62, "Sm", "samarium",        1.98,      150.36, 1.85,     0.0),
    ( 63, "Eu", "europium",        1.98,     151.964, 1.83,     0.0),
    ( 64, "Gd", "gadolinium",      1.96,      157.25, 1.82,     0.0),
    ( 65, "Tb", "terbium",         1.94,   158.92535, 1.81,     0.0),
    ( 66, "Dy", "dysprosium",      1.92,       162.5,  1.8,     0.0),
    ( 67, "Ho", "holmium",         1.92,   164.93033, 1.79,     0.0),
    ( 68, "Er", "erbium",          1.89,     167.259, 1.77,     0.0),
    ( 69, "Tm", "thulium",          1.9,   168.93422, 1.77,     0.0),
    ( 70, "Yb", "ytterbium",       1.87,     173.054, 1.78,     0.0),
    ( 71, "Lu", "lutetium",        1.87,    174.9668, 1.74,     0.0),
    ( 72, "Hf", "hafnium",         1.75,      178.49, 1.64,     0.0),
    ( 73, "Ta", "tantalum",         1.7,   180.94788, 1.58,     0.0),
    ( 74, "W",  "tungsten",        1.62,      183.84,  1.5,     0.0),
    ( 75, "Re", "rhenium",         1.51,     186.207, 1.41,     0.0),
    ( 76, "Os", "osmium",          1.44,      190.23, 1.36,     0.0),
    ( 77, "Ir", "iridium",         1.41,     192.217, 1.32,     0.0),
    ( 78, "Pt", "platinum",        1.36,     195.084,  1.3,     0.0),
    ( 79, "Au", "gold",            1.36,  196.966569,  1.3,     0.0),
    ( 80, "Hg", "mercury",         1.32,     200.592, 1.32,     0.0),
    ( 81, "Tl", "thallium",        1.45,      204.38, 1.44, -0.0255),
    ( 82, "Pb", "lead",            1.46,       207.2, 1.45, -0.0277),
    ( 83, "Bi", "bismuth",         1.48,    208.9804,  1.5, -0.0265),
    ( 84, "Po", "polonium",         1.4,         209, 1.42, -0.0198),
    ( 85, "At", "astatine",         1.5,         210, 1.48, -0.0155),
    ( 86, "Rn", "radon",            1.5,         222, 1.46, -0.0269),
    ( 87, "Fr", "francium",        None,         223, 2.42,  0.0046),
    ( 88, "Ra", "radium",          None,         226, 2.11,     0.0),
    ( 89, "Ac", "actinium",        None,         227, 2.01,     0.0),
    ( 90, "Th", "thorium",         None,    232.0377,  1.9,     0.0),
    ( 91, "Pa", "protactinium",    None,   231.03588, 1.84,     0.0),
    ( 92, "U",  "uranium",         1.96,   238.02891, 1.83,     0.0),
    ( 93, "Np", "neptunium",       None,         237,  1.8,     0.0),
    ( 94, "Pu", "plutonium",       None,         244,  1.8,     0.0),
    ( 95, "Am", "americium",       None,         243, 1.73,     0.0),
    ( 96, "Cm", "curium",          None,         247, 1.68,     0.0),
    ( 97, "Bk", "berkelium",       None,         247, 1.68,     0.0),
    ( 98, "Cf", "californium",     None,         251, 1.68,     0.0),
    ( 99, "Es", "einsteinium",     None,         252, 1.65,     0.0),
    (100, "Fm", "fermium",         None,         257, 1.67,     0.0),
    (101, "Md", "mendelevium",     None,         258, 1.73,     0.0),
    (102, "No", "nobelium",        None,         259, 1.76,     0.0),
    (103, "Lr", "lawrencium",      None,         262, 1.61,     0.0),
    (104, "Rf", "rutherfordium",   None,        None, 1.57,     0.0),
    (105, "Db", "dubnium",         None,        None, 1.49,     0.0),
    (106, "Sg", "seaborgium",      None,        None, 1.43,     0.0),
    (107, "Bh", "bohrium",         None,        None, 1.41,     0.0),
    (108, "Hs", "hassium",         None,        None, 1.34,     0.0),
    (109, "Mt", "meitnerium",      None,        None, 1.29,     0.0),
    (110, "Ds", "darmstadtium",    None,        None, 1.28,     0.0),
    (111, "Rg", "roentgenium",     None,        None, 1.21,     0.0),
    (112, "Cn", "copernicium",     None,        None, 1.22,     0.0),
    (113, "Nh", "nihonium",        None,        None, 1.36, -0.0179),
    (114, "Fl", "flerovium",       None,        None, 1.43, -0.0195),
    (115, "Mc", "moscovium",       None,        None, 1.62, -0.0187),
    (116, "Lv", "livermorium",     None,        None, 1.75,  -0.014),
    (117, "Ts", "tennessine",      None,        None, 1.65,  -0.011),
    (118, "Og", "oganesson",       None,        None, 1.57, -0.0189),
)
# fmt: on

_ELEMENTS = tuple(Element(name, symbol, z) for z, symbol, name, *_ in _ELEMENT_TABLE)

#: Element by lower-case symbol ("fe"), and by atomic number.
element_dict_atname = {element.symbol.lower(): element for element in _ELEMENTS}
element_dict_atnum = {element.atomnumber: element for element in _ELEMENTS}

#: Atomic number by lower-case symbol.
elematomnumbers = {symbol.lower(): z for z, symbol, *_ in _ELEMENT_TABLE}

#: Standard atomic weights, indexed by Z-1. Covers Z=1 (H) to Z=103 (Lr).
atommasses = [row[4] for row in _ELEMENT_TABLE[1:104]]

#: Covalent radii in Angstrom by capitalised symbol ("Fe"), for elements that have one.
eldict_covrad = {symbol: radius for _, symbol, _, radius, *_ in _ELEMENT_TABLE if radius is not None}

#: CM5 parameters, indexed by Z-1 as calc_cm5 expects. Cover Z=1 to Z=118.
cm5_radii = np.array([row[5] for row in _ELEMENT_TABLE[1:]])
cm5_dz = np.array([row[6] for row in _ELEMENT_TABLE[1:]])


# fmt: off
_ATOM_TYPES_BY_ELEMENT = {
    "H": (
        "H", "HA", "HB", "HW", "HH", "HN", "HD", "HZ", "HG", "HE", "HT", "H1", "H2", "H3", "HT1", "HT2", "HT3",
        "HC", "HF", "HN1", "HN2", "HB1", "HB2", "HG1", "HG2", "HG11", "HG12", "HG13", "HG21", "HG22", "HG23",
        "HH11", "HH12", "HH21", "HH22", "HD11", "HD12", "HD13", "HD21", "HD22", "HD23", "HE21", "HE22", "HE1",
        "HE2", "HE3", "HD1", "HD2", "1HH1", "2HH1", "1HH2", "2HH2", "1HG1", "2HG1", "3HG1", "1HG2", "2HG2",
        "3HG2", "1HD1", "2HD1", "3HD1", "1HD2", "2HD2", "3HD2", "H4", "HD3", "1HE2", "2HE2", "HZ1", "HZ2", "HZ3",
        "HB3", "HA1", "HA2", "HA3", "HH2", "HOB", "HW1", "HW2", "HO1", "H5", "H61", "HO", "H62", "HO6", "H12",
        "H52", "H22", "HO22", "H32", "HO32", "H631", "H632", "H633", "H43", "HO43", "H33", "HO33", "H23", "HO23",
        "H53", "HO53", "H42", "HO42", "H621", "H622", "HO62", "H13", "H01", "H02", "H03", "H04", "H05", "H06",
        "H07", "H08", "H09", "H0A", "H0B"
    ),
    "Na": ("NA", "SOD"),
    "C": (
        "C", "CD", "CH", "CH2", "CA", "CB", "CG", "CZ", "CE", "CG1", "CG2", "CD1", "CD2", "CE1", "CE2", "CE3",
        "C5", "C3", "C6", "CZ3", "CZ2", "CX", "CBC", "CAC", "CDC", "C1", "C2", "CT", "C4", "C12", "C52", "C22",
        "C32", "C42", "C62", "C13", "C53", "C23", "C33", "C43", "C63", "C00", "C01", "C02", "C03", "C04"
    ),
    "S": ("S", "SD", "SG", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9"),
    "O": (
        "O", "OE", "OG", "OH", "OW", "OT", "OD", "OP", "OE1", "O2", "OE2", "OD1", "OD2", "OG1", "OT1", "OH2",
        "OXT", "O1", "O5", "O04", "OT2", "OHB", "OB1", "OB2", "OA1", "OA2", "O3", "O4", "O6", "O52", "O22", "O32",
        "O42", "O62", "O53", "O23", "O33", "O43"
    ),
    "N": (
        "N", "NZ", "NH", "NE", "ND", "NH1", "NH2", "NE1", "NE2", "ND1", "ND2", "N1", "N01", "N02", "N03", "N04",
        "N05"
    ),
    "Cl": ("CL", "CLA"),
    "Mg": ("MG",),
    "Fe": ("FE1", "FE2", "FE3", "FE4", "FE5", "FE6", "FE7", "FE8", "FE9"),
    "Mo": ("MO1",),
    "V": ("V1",),
    "M": ("MW",),
}
# fmt: on

#: Element symbol by atom type or atom name.
atomtypes_dict = {atomtype: element for element, atomtypes in _ATOM_TYPES_BY_ELEMENT.items() for atomtype in atomtypes}

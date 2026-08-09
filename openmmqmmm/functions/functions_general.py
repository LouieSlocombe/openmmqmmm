import numpy as np
import os
import sys
import time

import openmmqmmm.settings_ash

# ANSI colors: http://jafrog.com/2013/11/23/colors-in-terminal.html
if openmmqmmm.settings_ash.settings_dict["use_ANSI_color"] is True:
    class BC:
        HEADER = '\033[95m'
        OKBLUE = '\033[94m'
        OKGREEN = '\033[92m'
        OKMAGENTA = '\033[95m'
        OKRED = '\033[31m'
        WARNING = '\033[93m'
        FAIL = '\033[91m'
        END = '\033[0m'
        BOLD = '\033[1m'
        UNDERLINE = '\033[4m'
else:
    class BC:
        HEADER = ''
        OKBLUE = ''
        OKGREEN = ''
        OKMAGENTA = ''
        OKRED = ''
        WARNING = ''
        FAIL = ''
        END = ''
        BOLD = ''
        UNDERLINE = ''




def is_interactive() -> bool:
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True  # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return True  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


# General function to exit ASH
# NOTE: By default we exit with errorcode 1
def ashexit(errormessage=None, code=1):
    print(BC.FAIL, "ASH exiting with code:", code, BC.END)

    if errormessage != None:
        print(BC.FAIL, "Error message:", errormessage, BC.END)

    # If in Jupyter notebook, then we do a softer return
    if is_interactive():
        raise SystemExit("ASH exiting due to error")
    else:
        sys.exit(1)


def basename(filename):
    return os.path.splitext(filename)[0]


# Grep-style function to find a line in file and return a list of words
# TODO: Make more advanced
def pygrep(string, file, errors=None):
    with open(file, errors=errors) as f:
        for line in f:
            if string in line:
                stringlist = line.split()
                return stringlist


# Multiple match version. Replace pygrep ?
def pygrep2(string, file, print_output=False, errors=None):
    l = []
    with open(file, errors=errors) as f:
        for line in f:
            if string in line:
                l.append(line)
    if print_output is True:
        print(*l)
    return l


# Simple function to do find and replace string in file
def find_replace_string_in_file(file, findstring, replstring):
    with open(file, 'r') as f:
        filedata = f.read()
    # Replace the target string
    filedata = filedata.replace(findstring, replstring)
    # Write the file out again
    with open(file, 'w') as f:
        f.write(filedata)


# Give difference of two lists, sorted. List1: Bigger list
def listdiff(list1, list2):
    diff = (list(set(list1) - set(list2)))
    diff.sort()
    return diff


# Print string if printlevel equals or larger than reference
def print_if_level(var, printlevel, refprintlevel):
    if printlevel >= refprintlevel:
        print(var)


# Debug print. Behaves like print but reads global debug var first
def printdebug(string, var=''):
    if openmmqmmm.settings_ash.settings_dict["debugflag"] is True:
        print(BC.OKRED, string, var, BC.END)


# mainmodule header
def print_line_with_mainheader(line):
    length = len(line)
    offset = 12
    outer_line = f"{BC.OKGREEN}{'#' * (length + offset)}{BC.END}"
    midline = f"{BC.OKGREEN}#{' ' * (length + offset - 2)}#{BC.END}"
    inner_line = f"{BC.OKGREEN}#{' ' * (offset // 2 - 1)}{BC.BOLD}{line}{' ' * (offset // 2 - 1)}#{BC.END}"
    print("\n")
    print(outer_line.center(80))
    print(midline.center(80))
    print(inner_line.center(80))
    print(midline.center(80))
    print(outer_line.center(80))


# Submodule header
def print_line_with_subheader1(line):
    print("")
    print(f"{BC.OKBLUE}{'-' * 80}{BC.END}")
    print(f"{BC.OKBLUE}{BC.BOLD}{line.center(80)}{BC.END}")
    print(f"{BC.OKBLUE}{'-' * 80}{BC.END}")
    print("")


# Submodule header
def print_line_with_subheader1_end():
    print("")
    print(f"{BC.OKBLUE}{'-' * 80}{BC.END}")


# Smaller header
def print_line_with_subheader2(line):
    print("")
    length = len(line)
    print(f"{BC.OKBLUE}{'-' * length}{BC.END}")
    print(f"{BC.OKBLUE}{BC.BOLD}{line}{BC.END}")
    print(f"{BC.OKBLUE}{'-' * length}{BC.END}")


# Inserts line into file for matched string.
# option: Once=True means only added for first match
def insert_line_into_file(file, string, addedstring, Once=True):
    Added = False
    with open(file, 'r') as ffr:
        contents = ffr.readlines()
    with open(file, 'w') as ffw:
        for l in contents:
            ffw.write(l)
            if string in l:
                if Added is False:
                    ffw.write(addedstring + '\n')
                    if Once is True:
                        Added = True


def blankline():
    print("")


# Can variable be converted into integer
def isint(s):
    try:
        int(s)
        return True
    except ValueError:
        return False
    except TypeError:
        return False


# Search list of lists. Returns list-index if match

def search_list_of_lists_for_index(i, l):
    return next((c for c, f in enumerate(l) if i in f), None)


# convert list of lists to dict
def create_conn_dict(l):
    index = {}
    for c, sublist in enumerate(l):
        for value in sublist:
            if value not in index:
                index[value] = c
    return index


# Read list of integers from file. Output list of integers. Ignores blanklines, return chars, non-int characters
# offset option: shifts integers by a value (e.g. 1 or -1)
def read_intlist_from_file(filename, offset=0):
    intlist = []
    try:
        with open(filename, "r") as f:
            for line in f:
                for l in line.split():
                    # Removing non-numeric part
                    l = ''.join(i for i in l if i.isdigit())
                    if isint(l):
                        intlist.append(int(l) + offset)
    except FileNotFoundError:
        print(f"File '{filename}' does not exists!")
        ashexit()
    intlist.sort()
    return intlist


# Write a string to file simply
def writestringtofile(string, file, writemode='w'):
    with open(file, writemode) as f:
        f.write(string)


# Write a Python list to file simply
def writelisttofile(pylist, file, separator=" "):
    with open(file, 'w') as f:
        for l in pylist:
            f.write(str(l) + separator)
    print("Wrote list to file:", file)


# Natural (human) sorting of list
def natural_sort(l):
    import re
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


# Reverse read function.


def clean_number(number):
    return np.real_if_close(number)


# Extract column from matrix
def column(matrix, i):
    return [row[i] for row in matrix]


# Various function to print time of module/step. Will add time also to Timings object
# Printing if currprintlevel
def print_time_rel(timestamp, modulename='Unknown', moduleindex=4, currprintlevel=1, currthreshold=1):
    secs = time.time() - timestamp
    mins = secs / 60
    if currprintlevel >= currthreshold:
        print_line_with_subheader2(
            "Time to calculate step ({}): {:4.3f} seconds, {:3.1f} minutes.".format(modulename, secs, mins))
    # Adding time to Timings object
    timingsobject.add(modulename, secs, moduleindex=moduleindex)


def print_time_tot_color(time_initial, modulename='Unknown', moduleindex=4):
    secs = time.time() - time_initial
    mins = secs / 60
    print(BC.WARNING, "-------------------------------------------------------------------", BC.END)
    print(BC.WARNING, "ASH Total Walltime: {:3.1f} seconds, {:3.1f} minutes.".format(secs, mins), BC.END)
    print(BC.WARNING, "-------------------------------------------------------------------", BC.END)
    # Adding time to Timings object
    timingsobject.add(modulename, secs, moduleindex=moduleindex)


# Keep track of module runtimes
class Timings:
    def __init__(self):
        self.simple_dict = {}
        self.module_count = {}
        self.module_indices = {}
        self.totalsumtime = 0

    def add(self, modulename, mtime, moduleindex=4):

        # Adding time to dictionary
        if modulename in self.simple_dict:
            self.simple_dict[modulename] += mtime
        else:
            self.simple_dict[modulename] = mtime

        # Adding moduleindex to dictionary
        if modulename not in self.module_indices:
            self.module_indices[modulename] = moduleindex

        # Adding times called
        if modulename in self.module_count:
            self.module_count[modulename] += 1
        else:
            self.module_count[modulename] = 1

        self.totalsumtime += mtime

    # Distinguish and sort between:
    # workflows (thermochem_protol, PES, calc_surface etc.): 0
    # jobtype (optimizer,Singlepoint,Anfreq,Numfreq): 1
    # theory-run (ORCAtheory run, QM/MM run, MM run etc.): 2
    # others (calc connectivity etc.): 4

    def print(self, inittime):
        totalwalltime = time.time() - inittime
        print("To turn off timing output add to settings file: ~/ash_user_settings.ini")
        print("print_full_timings = False   ")
        print("")
        print("{:35}{:>20}{:>20}{:>17}".format("Modulename", "Time (sec)", "Percentage of total", "Times called"))
        print("-" * 100)

        # Lists of dictitems by module_labels
        # Workflows: thermochemprotocol, calc_surface, benchmarking etc.
        dictitems_index0 = [i for i in self.simple_dict if self.module_indices[i] == 0]
        # Jobtype: Singlepoint, Opt, freq
        dictitems_index1 = [i for i in self.simple_dict if self.module_indices[i] == 1]
        # Theory run: ORCATHeory, QM/MM Theory etc
        dictitems_index2 = [i for i in self.simple_dict if self.module_indices[i] == 2]
        # NOTE: Was not using index 3. Now using for object creation
        dictitems_index3 = [i for i in self.simple_dict if self.module_indices[i] == 3]
        # Other small modules. 4 is default
        dictitems_index4 = [i for i in self.simple_dict if self.module_indices[i] == 4]

        if len(dictitems_index0) != 0:
            print("Workflow modules")
            print("-" * 30)
            for dictitem in dictitems_index0:
                mmtime = self.simple_dict[dictitem]
                time_per = 100 * (mmtime / totalwalltime)
                print("{:35}{:>20.2f}{:>10.1f}{:>20}".format(dictitem, mmtime, time_per, self.module_count[dictitem]))
            print("")
        if len(dictitems_index1) != 0:
            print("Jobtype modules")
            print("-" * 30)
            for dictitem in dictitems_index1:
                mmtime = self.simple_dict[dictitem]
                time_per = 100 * (mmtime / totalwalltime)
                print("{:35}{:>20.2f}{:>10.1f}{:>20}".format(dictitem, mmtime, time_per, self.module_count[dictitem]))
            print("")
        if len(dictitems_index2) != 0:
            print("Theory-run modules")
            print("-" * 30)
            for dictitem in dictitems_index2:
                mmtime = self.simple_dict[dictitem]
                time_per = 100 * (mmtime / totalwalltime)
                print("{:35}{:>20.2f}{:>10.1f}{:>20}".format(dictitem, mmtime, time_per, self.module_count[dictitem]))
            print("")
        if len(dictitems_index3) != 0:
            print("Object creation")
            print("-" * 30)
            for dictitem in dictitems_index3:
                mmtime = self.simple_dict[dictitem]
                time_per = 100 * (mmtime / totalwalltime)
                print("{:35}{:>20.2f}{:>10.1f}{:>20}".format(dictitem, mmtime, time_per, self.module_count[dictitem]))
            print("")
        if len(dictitems_index4) != 0:
            print("Other modules")
            print("-" * 30)
            for dictitem in dictitems_index4:
                mmtime = self.simple_dict[dictitem]
                time_per = 100 * (mmtime / totalwalltime)
                print("{:35}{:>20.2f}{:>10.1f}{:>20}".format(dictitem, mmtime, time_per, self.module_count[dictitem]))
            print("")
        print("")
        print("{:35}{:>20.2f}".format("Sum of all moduletimes (flawed)", self.totalsumtime))
        print("{:35}{:>20.2f}{:>10}".format("Total walltime", totalwalltime, 100.0))


# Creating object
timingsobject = Timings()

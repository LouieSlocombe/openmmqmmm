"""
Functions to print header, footer, inputscript etc.
"""
import os
import sys
import time

import openmmqmmm.settings_ash
from openmmqmmm.functions.functions_general import ashexit, BC, print_time_tot_color, timingsobject, \
    print_line_with_subheader1

try:
    from importlib.metadata import version
    programversion = version("openmmqmmm")
except Exception:
    programversion = "unknown"


# ASH footer
def print_footer():
    print()
    print_time_tot_color(init_time)


def print_timings():
    """
    Print timings of each module
    """
    print()
    print_line_with_subheader1("Total timings of all modules")

    timingsobject.print(init_time)


def print_header():
    """
    Initial output: header (name, version), initial time, settings, inputscript.
    """

    # Initializes time
    global init_time
    init_time = time.time()

    print(f"{BC.OKGREEN}{'-' * 80}{BC.END}")
    print(f"{BC.OKGREEN}{'-' * 80}{BC.END}")
    print("openmmqmmm".center(90))
    print(f"{BC.WARNING}ORCA + OpenMM QM/MM (trimmed ASH distribution){BC.END}".center(90))
    print(f"{BC.WARNING}{BC.BOLD}Version: {programversion}{BC.END}".center(95))
    print(f"{BC.OKGREEN}{'-' * 80}{BC.END}")
    print(f"{BC.OKGREEN}{'-' * 80}{BC.END}")

    print("Package path:", openmmqmmm.settings_ash.ashpath)

    # Check Python version
    pythonversion = (sys.version_info[0], sys.version_info[1], sys.version_info[2])
    print("Python version: {}.{}.{}".format(pythonversion[0], pythonversion[1], pythonversion[2]))
    print("Python interpreter:", sys.executable)
    if pythonversion < (3, 10, 0):
        print("openmmqmmm requires Python version 3.10.0 or higher")
        ashexit()

    print("\nSettings after reading defaults and ~/ash_user_settings.ini : ")
    for key, val in openmmqmmm.settings_ash.settings_dict.items():
        print("\t", key, ": ", val)

    print("\nNote: ANSI escape sequences can be used for displaying color. Use e.g. less -R to display")
    print("To turn on/off escape sequences, set: 'use_ANSI_color = False' in")
    print("~/ash_user_settings.ini")
    print()

    # Print input script unless interactive session or pytest
    if openmmqmmm.settings_ash.settings_dict["print_input"] is True:
        if openmmqmmm.settings_ash.interactive_session is False:
            # Ignore if pytest is active
            if "pytest" not in sys.modules:
                try:
                    inputfilepath = os.getcwd() + "/" + sys.argv[0]
                    print("Input script:", inputfilepath)
                    print(f"{BC.WARNING}{'=' * 80}")
                    with open(inputfilepath) as f:
                        for line in f:
                            print("   >", line, end="")
                except Exception:
                    pass
                print(f"{BC.WARNING}{'=' * 80}", BC.END)
                print()

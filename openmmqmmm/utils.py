import logging
import os
import re
import time

import numpy as np

from openmmqmmm.exceptions import (
    FileFormatError,
)

logger = logging.getLogger(__name__)
timings_logger = logging.getLogger("openmmqmmm.timings")


def configure_logging(level="INFO", file=None, fmt="%(message)s") -> logging.Logger:
    """Configure output for openmmqmmm calculations."""
    package_logger = logging.getLogger("openmmqmmm")
    env_level = os.environ.get("OPENMMQMMM_LOGLEVEL")
    if env_level:
        level = env_level
    package_logger.setLevel(level.upper() if isinstance(level, str) else level)
    formatter = logging.Formatter(fmt)
    # Replace handlers configured by a previous call rather than stacking them
    for handler in list(package_logger.handlers):
        if getattr(handler, "_openmmqmmm_handler", False):
            package_logger.removeHandler(handler)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler._openmmqmmm_handler = True
    package_logger.addHandler(stream_handler)
    if file is not None:
        file_handler = logging.FileHandler(file)
        file_handler.setFormatter(formatter)
        file_handler._openmmqmmm_handler = True
        package_logger.addHandler(file_handler)
    return package_logger


def log_time_since(timestamp, label="step"):
    secs = time.time() - timestamp
    timings_logger.debug("Time to calculate step (%s): %.3f seconds, %.1f minutes", label, secs, secs / 60)


def main_header(text):
    width = len(text) + 12
    edge = "#" * width
    mid = "#" + " " * (width - 2) + "#"
    inner = "#" + text.center(width - 2) + "#"
    return "\n".join(["\n", edge.center(80), mid.center(80), inner.center(80), mid.center(80), edge.center(80)])


def sub_header(text):
    rule = "-" * 80
    return f"\n{rule}\n{text.center(80)}\n{rule}\n"


def sub_header_end():
    return "\n" + "-" * 80


def small_header(text):
    rule = "-" * len(text)
    return f"\n{rule}\n{text}\n{rule}"


def basename(filename):
    return os.path.splitext(filename)[0]


def pygrep(string, file, errors=None):
    with open(file, errors=errors) as f:
        for line in f:
            if string in line:
                return line.split()
    return None


def pygrep2(string, file, print_output=False, errors=None):
    with open(file, errors=errors) as f:
        matches = [line for line in f if string in line]
    if print_output is True:
        logger.info("%s", "".join(matches))
    return matches


# Simple function to do find and replace string in file
def find_replace_string_in_file(file, findstring, replstring):
    with open(file) as f:
        filedata = f.read()
    # Replace the target string
    filedata = filedata.replace(findstring, replstring)
    # Write the file out again
    with open(file, "w") as f:
        f.write(filedata)


# Give difference of two lists, sorted. List1: Bigger list
def listdiff(list1, list2):
    diff = list(set(list1) - set(list2))
    diff.sort()
    return diff


# Inserts line into file for matched string.
# option: Once=True means only added for first match
def insert_line_into_file(file, string, addedstring, once=True):
    added = False
    with open(file) as ffr:
        contents = ffr.readlines()
    with open(file, "w") as ffw:
        for content_line in contents:
            ffw.write(content_line)
            if string in content_line and added is False:
                ffw.write(addedstring + "\n")
                if once is True:
                    added = True


# Can variable be converted into integer
def isint(s):
    try:
        int(s)
        return True
    except ValueError:
        return False
    except TypeError:
        return False


def search_list_of_lists_for_index(i, list_of_lists):
    return next((c for c, f in enumerate(list_of_lists) if i in f), None)


# convert list of lists to dict
def create_conn_dict(list_of_lists):
    index = {}
    for c, sublist in enumerate(list_of_lists):
        for value in sublist:
            if value not in index:
                index[value] = c
    return index


# Read list of integers from file. Output list of integers. Ignores blanklines, return chars, non-int characters
# offset option: shifts integers by a value (e.g. 1 or -1)
def read_intlist_from_file(filename, offset=0):
    intlist = []
    try:
        with open(filename) as f:
            for line in f:
                for word in line.split():
                    # Removing non-numeric part
                    digits = "".join(i for i in word if i.isdigit())
                    if isint(digits):
                        intlist.append(int(digits) + offset)
    except FileNotFoundError:
        raise FileFormatError(f"File '{filename}' does not exists!") from None
    intlist.sort()
    return intlist


# Write a string to file simply
def writestringtofile(string, file, writemode="w"):
    with open(file, writemode) as f:
        f.write(string)


# Write a Python list to file simply
def writelisttofile(pylist, file, separator=" "):
    with open(file, "w") as f:
        f.writelines(str(item) + separator for item in pylist)
    logger.info("Wrote list to file: %s", file)


def natural_sort(items):
    def alphanum_key(key):
        return [int(part) if part.isdigit() else part.lower() for part in re.split("([0-9]+)", key)]

    return sorted(items, key=alphanum_key)


def clean_number(number):
    return np.real_if_close(number)


def column(matrix, i):
    return [row[i] for row in matrix]

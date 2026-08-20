from __future__ import annotations

import logging
import os
import re
import time
from collections.abc import Iterable, Sequence
from os import PathLike
from typing import Any, TypeVar

import numpy as np

from openmmqmmm.exceptions import (
    FileFormatError,
)

logger = logging.getLogger(__name__)
timings_logger = logging.getLogger("openmmqmmm.timings")


_T = TypeVar("_T")


def configure_logging(
    level: int | str = "INFO",
    file: str | PathLike[str] | None = None,
    fmt: str = "%(message)s",
) -> logging.Logger:
    """Configure console (and optional file) output; OPENMMQMMM_LOGLEVEL overrides level."""
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


def log_time_since(timestamp: float, label: str = "step") -> None:
    secs = time.time() - timestamp
    timings_logger.debug("Time to calculate step (%s): %.3f seconds, %.1f minutes", label, secs, secs / 60)


def main_header(text: str) -> str:
    width = len(text) + 12
    edge = "#" * width
    mid = "#" + " " * (width - 2) + "#"
    inner = "#" + text.center(width - 2) + "#"
    return "\n".join(["\n", edge.center(80), mid.center(80), inner.center(80), mid.center(80), edge.center(80)])


def sub_header(text: str) -> str:
    rule = "-" * 80
    return f"\n{rule}\n{text.center(80)}\n{rule}\n"


def sub_header_end() -> str:
    return "\n" + "-" * 80


def small_header(text: str) -> str:
    rule = "-" * len(text)
    return f"\n{rule}\n{text}\n{rule}"


def basename(filename: str | PathLike[str]) -> str:
    return os.path.splitext(filename)[0]


def pygrep(
    string: str,
    file: str | PathLike[str],
    errors: str | None = None,
) -> list[str] | None:
    with open(file, errors=errors) as f:
        for line in f:
            if string in line:
                return line.split()
    return None


def pygrep2(
    string: str,
    file: str | PathLike[str],
    print_output: bool = False,
    errors: str | None = None,
) -> list[str]:
    with open(file, errors=errors) as f:
        matches = [line for line in f if string in line]
    if print_output is True:
        logger.info("%s", "".join(matches))
    return matches


def find_replace_string_in_file(
    file: str | PathLike[str],
    findstring: str,
    replstring: str,
) -> None:
    with open(file) as f:
        filedata = f.read()
    filedata = filedata.replace(findstring, replstring)
    with open(file, "w") as f:
        f.write(filedata)


def listdiff(list1: Iterable[_T], list2: Iterable[_T]) -> list[_T]:
    diff = list(set(list1) - set(list2))
    diff.sort()
    return diff


def insert_line_into_file(
    file: str | PathLike[str],
    string: str,
    addedstring: str,
    once: bool = True,
) -> None:
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


def isint(s: object) -> bool:
    try:
        int(s)
        return True
    except ValueError:
        return False
    except TypeError:
        return False


def search_list_of_lists_for_index(i: _T, list_of_lists: Sequence[Sequence[_T]]) -> int | None:
    return next((c for c, f in enumerate(list_of_lists) if i in f), None)


def read_intlist_from_file(filename: str | PathLike[str], offset: int = 0) -> list[int]:
    intlist = []
    try:
        with open(filename) as f:
            for line in f:
                for word in line.split():
                    digits = "".join(i for i in word if i.isdigit())
                    if isint(digits):
                        intlist.append(int(digits) + offset)
    except FileNotFoundError:
        raise FileFormatError(f"File '{filename}' does not exists!") from None
    intlist.sort()
    return intlist


def write_string_to_file(
    string: str,
    file: str | PathLike[str],
    writemode: str = "w",
) -> None:
    with open(file, writemode) as f:
        f.write(string)


def write_list_to_file(
    pylist: Iterable[object],
    file: str | PathLike[str],
    separator: str = " ",
) -> None:
    with open(file, "w") as f:
        f.writelines(str(item) + separator for item in pylist)
    logger.info("Wrote list to file: %s", file)


def natural_sort(items: Iterable[str]) -> list[str]:
    def alphanum_key(key: str) -> list[int | str]:
        return [int(part) if part.isdigit() else part.lower() for part in re.split("([0-9]+)", key)]

    return sorted(items, key=alphanum_key)


def clean_number(number: Any) -> Any:
    return np.real_if_close(number)


def column(matrix: Iterable[Sequence[_T]], i: int) -> list[_T]:
    return [row[i] for row in matrix]

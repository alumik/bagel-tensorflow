import pathlib

from typing import *


def mkdirs(*dir_list):
    for directory in dir_list:
        pathlib.Path(directory).mkdir(parents=True, exist_ok=True)


def file_list(path: pathlib.Path) -> List[pathlib.Path]:
    if path.is_dir():
        return list(path.iterdir())
    return [path]

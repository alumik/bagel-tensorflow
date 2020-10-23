import os
import bagel
import pandas as pd

from typing import Sequence


def filename(file: str) -> str:
    return os.path.splitext(os.path.basename(file))[0]


def mkdirs(*dir_list):
    for directory in dir_list:
        if not os.path.isdir(directory):
            os.makedirs(directory)


def file_list(path: str) -> Sequence:
    if os.path.isdir(path):
        return [os.path.join(path, file) for file in os.listdir(path)]
    else:
        return [path]


def load_kpi(file: str, **kwargs) -> bagel.data.KPI:
    df = pd.read_csv(file, **kwargs)
    labels = df.label if 'label' in df.keys() else None
    return bagel.data.KPI(timestamps=df.timestamp,
                          values=df.value,
                          labels=labels,
                          name=bagel.utils.filename(file))

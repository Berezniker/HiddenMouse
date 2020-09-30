from utils.color import COLOR
import pandas as pd
import shutil
import sys
import os


def clear_directory(path: str) -> None:
    """
    Removes all content in the directory
    Confirmation required for deletion

    :param path: path to directory
    :return: None
    """
    if os.path.exists(path):
        print(f"{COLOR['boldFont']}{COLOR['red']}[Warning]: remove directory: {path}{COLOR['none']}")
        key = input("Do you want to continue? [Y/N]: ")
        if key.upper() == 'Y':
            shutil.rmtree(path)
        else:
            sys.exit(0)
    os.makedirs(path)


def check_min_size(db: pd.DataFrame, min_size: int = 1000) -> bool:
    """
    Checker

    :param db: database
    :param min_size: lower size bound
    :return: <True> if database size is less than <min_size>,
             otherwise <False>
    """
    return db.index.size < min_size


# https://en.wikipedia.org/wiki/Quartile#Outliers
def quartile(db: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Replace the values in <db[col]> outside the IQR
    with the upper and lower boundaries, respectively.

    :param db: database
    :param col: column name
    :return: transformed database
    """
    _, bins = pd.qcut(x=db[col],
                      q=[0.25, 0.75],  # [0.0, 0.25, 0.5, 0.75, 1.0],
                      retbins=True, duplicates='drop')
    q1 = bins[0]  # lower (first) quartile
    q3 = bins[1]  # upper (third) quartile
    iqr = q3 - q1  # InterQuartile Range
    lower_fence = (q1 - 1.5 * iqr)
    upper_fence = (q3 + 1.5 * iqr)
    db.loc[db[col] < lower_fence, col] = q1
    db.loc[db[col] > upper_fence, col] = q3
    return db


def fix_duplicate_time(db: pd.DataFrame) -> pd.DataFrame:
    """
    Replace duplicated timestamps with average neighbors` values

    :param db: database
    :return: transformed database
    """
    db.reset_index(drop=True, inplace=True)
    n_dup = -1
    while (db.time.diff() == 0).any():
        idx = db[db.time.diff() == 0].index
        if n_dup == idx.size:
            db.drop_duplicates(subset='time', inplace=True)
            break
        n_dup = idx.size
        db.loc[idx, 'time'] = (db.reindex(idx - 1).time.values +
                               db.reindex(idx + 1).time.values) / 2.0
    db.dropna(inplace=True)  # sometimes the last lines contains nan
    return db


def add_differential_fields(db: pd.DataFrame) -> pd.DataFrame:
    """
    Add <dx>, <dy>, <dt>

    :param db: database
    :return: transformed database
    """
    db['dx'] = db.x.diff()
    db['dy'] = db.y.diff()
    db['dt'] = db.time.diff()

    db.loc[db.dt < 1e-3, 'dt'] = 1e-3

    return db


class Informant:
    def __init__(self, check: bool, size: int):
        self.print = print if check else lambda *args, **kwargs: None
        self.size = size
        self.prev_percent = 0.0
        self.print(f"    database.size = {self.size:6}", end=" ")

    def __call__(self, action: str, size: int):
        percent = (1.0 - size / self.size) * 100.0 - self.prev_percent
        self.prev_percent += percent
        percent = f"{COLOR['yellow']}-{round(percent, 1):4}%{COLOR['none']}"
        self.print(f"--({action})--> {size:6} [{percent}]", end=" ")

    def __del__(self):
        self.print(f"{COLOR['yellow']} -{round(self.prev_percent, 1):4}%{COLOR['none']}")


def preprocessing(database: pd.DataFrame,
                  check_size: bool = False) -> pd.DataFrame:
    """
    Full preprocessing of raw data

    :param database: raw data
    :param check_size: if <True> print log in stdout ~ (verbose == 3)
    :return: transformed database
    """
    info = Informant(check_size, database.index.size)
    database['time'] = database['time'].round(3)

    database.drop_duplicates(inplace=True)
    info("drop_duplicates", database.index.size)

    database = fix_duplicate_time(database)
    info("fix_time_duplicates", database.index.size)

    database = quartile(quartile(database, 'x'), 'y')
    info("quartile", database.index.size)

    return database

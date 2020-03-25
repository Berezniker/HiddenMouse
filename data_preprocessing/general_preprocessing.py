from utils.color import COLOR
import pandas as pd
import numpy as np
import shutil
import os


def clear_directory(path: str) -> None:
    if os.path.exists(path):
        print(f"{COLOR['boldFont']}{COLOR['red']}[Warning]: remove directory: {path}{COLOR['none']}")
        key = input("Do you want to continue? [Y/N]: ")
        if key.upper() == 'Y':
            shutil.rmtree(path)
        else:
            exit(-1)


def preprocessing(database: pd.DataFrame,
                  inplace: bool = True,
                  check_size: bool = False) -> pd.DataFrame:
    if check_size:
        print(f"database.size = {database.index.size}", end=" ")
    database.drop_duplicates(inplace=inplace)
    if check_size:
        print(f"--(drop_duplicates)--> {database.index.size}", end=" ")
    database.drop_duplicates(subset='time', inplace=inplace)
    if check_size:
        print(f"--(drop_time_duplicates)--> {database.index.size}", end=" ")
    diff = (database[['x', 'y']] - database[['x', 'y']].shift(1))
    database.drop(database[np.all(diff == 0, axis=1)].index, inplace=inplace)
    if check_size:
        print(f"--(drop_xy_duplicates)--> {database.index.size}", end=" ")
    database.drop(database[database.x > 2000].index, inplace=inplace)
    database.drop(database[database.y > 1200].index, inplace=inplace)
    if check_size:
        print(f"--(drop_outliers)--> {database.index.size}")
    return database

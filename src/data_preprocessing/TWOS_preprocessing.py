from data_preprocessing.general_preprocessing import *
from utils.color import COLOR
from itertools import count
import pandas as pd
import numpy as np
import time
import glob
import os

######################################################################
# TWOS dataset
#    |
#    |-- time       <-- converted date to sec
#    |-- state      <-- deleted
#    |-- x: int
#    |-- y: int
#    |-- username   <-- deleted
#    |-- ?unknown?  <-- deleted
#    |-- ?unknown?  <-- deleted
######################################################################


def twos_preprocessing(data_dir: str = '../../original_dataset/TWOS_original/mouse_ano',
                       save_dir: str = '../../dataset/TWOS',
                       verbose: int = 0) -> None:
    uniq_name = count()
    clear_directory(save_dir)
    save_dir = os.path.join(save_dir, 'train_files')
    header = ['time', 'state', 'x', 'y', 'username', 'u1', 'u2']
    drop_fields = ['state', 'username', 'u1', 'u2']

    for user_path in glob.glob(os.path.join(data_dir, 'User*')):
        user_name = os.path.basename(user_path).split('.')[0].lower()
        if verbose >= 2: print(f"\n>> {user_name}")
        # PREPROCESSING
        db = pd.read_csv(user_path, sep=';', names=header, parse_dates=[0])
        db.drop(drop_fields, axis=1, inplace=True)
        db.dropna(inplace=True)
        db.time = (db.time - db.time.iloc[0]).dt.total_seconds()
        db.x, db.y = db.x.astype(int), db.y.astype(int)
        db.reset_index(drop=True, inplace=True)
        db = preprocessing(db, check_size=(verbose >= 3))
        if check_min_size(db): continue

        # PRESERVATION
        train_save_path = os.path.join(save_dir, user_name)
        test_save_path = train_save_path.replace('train', 'test')
        os.makedirs(train_save_path, exist_ok=True)
        os.makedirs(test_save_path, exist_ok=True)
        train_save_path = os.path.join(train_save_path, f"session_{next(uniq_name):02}.csv")
        test_save_path = os.path.join(test_save_path, f"session_{next(uniq_name):02}.csv")

        train_db = db[:int(db.index.size * 0.7)].copy()
        test_db = db[-int(db.index.size * 0.3):].copy()
        test_db.time -= test_db.time.iloc[0]
        train_db.to_csv(train_save_path, index=False)
        test_db.to_csv(test_save_path, index=False)


if __name__ == '__main__':
    start_time = time.time()
    print(f"{COLOR['italics']}TWOS{COLOR['none']} Run!")
    twos_preprocessing(verbose=3)
    print(f"run time: {time.time() - start_time:.3f}")

from data_preprocessing.general_preprocessing import *
from utils.color import COLOR
from itertools import count
import utils.constants as const
import pandas as pd
import time
import glob
import os


# [link](https://ms.sapientia.ro/~manyi/DFL.html)

######################################################################
# DFL dataset
#    |
#    |-- client timestamp (in sec): float  <-- renamed: 'time'
#    |-- button: ['NoButton', 'Left', 'Scroll', 'Right']  <-- deleted
#    |-- state:  ['Move', 'Pressed', 'Released', 'Drag', 'Down', 'Up']
#    |     ^--- deleted
#    |-- x: int
#    |-- y: int
######################################################################


def dfl_preprocessing(verbose: int = 0) -> None:
    """
    DFL data preprocessing

    :param verbose: verbose output to stdout,
                    0 -- silence, [1, 2, 3] -- more verbose
    :return: None
    """
    data_dir = os.path.join(const.ORIGINAL_DATASET_PATH, "DFL_original")
    save_dir = os.path.join(const.DATASET_PATH, "DFL")
    rename_fields = {'client timestamp': 'time'}
    drop_fields = ['button', 'state']
    clear_directory(save_dir)
    save_dir = os.path.join(save_dir, 'train_files')
    uniq_name = count()

    for user_path in glob.glob(os.path.join(data_dir, 'User*')):
        user_name = os.path.basename(user_path).lower()
        user_name = f"{user_name[:4]}{user_name[4:]:>02}"
        if verbose >= 2: print(f"\n>> {user_name}")
        for session_path in glob.glob(os.path.join(user_path, '*.CSV')):
            session_name = os.path.basename(session_path).lower()
            if verbose >= 3: print(f">>> {session_name}", end='')
            # PREPROCESSING
            db = pd.read_csv(session_path)
            db.rename(rename_fields, axis=1, inplace=True)
            db.drop(drop_fields, axis=1, inplace=True)
            db.time = (db.time - db.time.iloc[0]) / 1000.0
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
    print(f"{COLOR['italics']}DFL{COLOR['none']} Run!")
    dfl_preprocessing(verbose=3)
    print(f"run time: {(time.time() - start_time) / 60.0:.1f} min")

from data_preprocessing.general_preprocessing import *
from utils.color import COLOR
import pandas as pd
import time
import glob
import os

######################################################################
# Balabit dataset
#    |
#    |-- record timestamp (in sec): float  <-- deleted
#    |-- client timestamp (in sec): float  <-- renamed: 'time'
#    |-- button: ['NoButton', 'Left', 'Scroll', 'Right']  <-- deleted
#    |-- state:  ['Move', 'Pressed', 'Released', 'Drag', 'Down', 'Up']
#    |     ^--- deleted
#    |-- x: int
#    |-- y: int
######################################################################


def balabit_preprocessing(data_dir: str = "../../original_dataset/BALABIT_original",
                          save_dir: str = "../../dataset/BALABIT",
                          verbose: int = 0) -> None:
    labels_path = os.path.join(data_dir, 'labels.csv')
    rename_fields = {'client timestamp': 'time'}
    drop_fields = ['record timestamp', 'button', 'state']

    labels = pd.read_csv(labels_path)

    clear_directory(save_dir)

    for path in glob.glob(os.path.join(data_dir, '*files')):
        if verbose >= 1: print(f"> {os.path.basename(path)}")
        for user_path in glob.glob(os.path.join(path, 'user*')):
            user_name = os.path.basename(user_path)
            if verbose >= 2: print(f">> {user_name}")
            for session_path in glob.glob(os.path.join(user_path, 'session*')):
                session_name = os.path.basename(session_path)
                check_illegal = labels[labels.filename == session_name]
                if check_illegal.size and check_illegal.is_illegal.values:
                    if verbose >= 3:
                        print(f"{COLOR['yellow']}Illegal session skipped: {session_name}{COLOR['none']}")
                    continue
                if os.path.exists(session_path):
                    db = pd.read_csv(session_path)
                    db.rename(rename_fields, axis=1, inplace=True)
                    db.drop(db[db.button == 'Scroll'].index, axis=0, inplace=True)
                    db.drop(drop_fields, axis=1, inplace=True)
                    preprocessing(db)
                    save_path = user_path.replace('original_dataset', 'dataset')
                    save_path = save_path.replace('BALABIT_original', 'BALABIT')
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    save_path = os.path.join(save_path, os.path.basename(session_path))
                    db.to_csv(save_path, index=False)


if __name__ == '__main__':
    start_time = time.time()
    print(f"{COLOR['italics']}BALABIT{COLOR['none']} Run!")
    balabit_preprocessing(verbose=2)
    print(f"run time: {time.time() - start_time:.3f}")

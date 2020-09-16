from data_preprocessing.general_preprocessing import *
from utils.color import COLOR
import pandas as pd
import time
import glob
import os


# [link](https://github.com/balabit/Mouse-Dynamics-Challenge)

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
        file_type = os.path.basename(path)
        if verbose >= 1: print(f"\n> {file_type}")
        for user_path in glob.glob(os.path.join(path, 'user*')):
            user_name = os.path.basename(user_path)
            if verbose >= 2: print(f">> {user_name}")
            for session_path in glob.glob(os.path.join(user_path, 'session*')):
                session_name = os.path.basename(session_path)
                check_illegal = labels[labels.filename == session_name]
                if check_illegal.size and check_illegal.is_illegal.values:
                    if verbose >= 3:
                        print(f"    {COLOR['yellow']}Illegal session skipped: {session_name}{COLOR['none']}")
                    continue
                # PREPROCESSING
                db = pd.read_csv(session_path)
                db.rename(rename_fields, axis=1, inplace=True)
                db.drop(db[db.button == 'Scroll'].index, axis=0, inplace=True)
                db.drop(drop_fields, axis=1, inplace=True)
                db = preprocessing(db, check_size=(verbose >= 3))
                if check_min_size(db, min_size=200):
                    if verbose >= 3:
                        print(f"    {COLOR['yellow']}Short   session skipped: {session_name}{COLOR['none']}")
                    continue

                # PRESERVATION
                save_path = os.path.join(save_dir, file_type)
                save_path = os.path.join(save_path, user_name)
                os.makedirs(save_path, exist_ok=True)
                save_path = os.path.join(save_path, session_name + ".csv")
                db.to_csv(save_path, index=False)


if __name__ == '__main__':
    start_time = time.time()
    print(f"{COLOR['italics']}BALABIT{COLOR['none']} Run!")
    balabit_preprocessing(verbose=3)
    print(f"run time: {(time.time() - start_time) / 60.0:.1f} min")

from data_preprocessing.general_preprocessing import *
from utils.color import COLOR
from itertools import count
import pandas as pd
import time
import glob
import os


######################################################################
# DATAIIT dataset
#    |
#    |-- time        <-- converted to format: sec
#    |-- username    <-- deleted
#    |-- processname <-- deleted
#    |-- message_id  <-- deleted
#    |-- x_pos: int  <-- renamed: 'x'
#    |-- y_pos: int  <-- renamed: 'y'
#    |-- record_info <-- deleted
#    |-- hwnd        <-- deleted
######################################################################


def dataiit_preprocessing(data_dir: str = '../../original_dataset/DATAIIT_original',
                          save_dir: str = '../../dataset/DATAIIT',
                          verbose: int = 0) -> None:
    rename_fields = {'x_pos': 'x', 'y_pos': 'y'}
    drop_fields = ['username', 'processname', 'message_id', 'record_info', 'hwnd']
    uniq_name = count()

    clear_directory(save_dir)
    save_dir = os.path.join(save_dir, 'train_files')

    for user_path in glob.glob(os.path.join(data_dir, 'User*')):
        user_name = os.path.basename(user_path).lower()
        if verbose >= 2: print(f"\n>> {user_name}")
        for session_path in glob.glob(os.path.join(user_path, f'*\\{user_name}*')):
            file_path = os.path.join(session_path, 'MOUSE.csv')
            if os.path.exists(file_path):
                # PREPROCESSING
                db = pd.read_csv(file_path, parse_dates=['time'], skiprows=[1])
                db.time = (db.time - db.time.iloc[0]).dt.total_seconds()
                db.rename(rename_fields, axis=1, inplace=True)
                db.drop(drop_fields, axis=1, inplace=True)
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
    print(f"{COLOR['italics']}DATAIIT{COLOR['none']} Run!")
    dataiit_preprocessing(verbose=3)
    print(f"run time: {(time.time() - start_time) / 60.0:.1f} min")

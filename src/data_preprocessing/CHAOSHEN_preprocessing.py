from data_preprocessing.general_preprocessing import *
from utils.color import COLOR
from itertools import count
import pandas as pd
import time
import glob
import os

######################################################################
# CHAOSHEN dataset
#    |
#    |-- code    <-- deleted
#    |-- x: int
#    |-- y: int
#    |-- time    <-- converted ms to sec
######################################################################


def chaoshen_preprocessing(data_dir: str = "../../original_dataset/CHAOSHEN_original/data",
                           save_dir: str = "../../dataset/CHAOSHEN",
                           verbose: int = 0) -> None:
    user_num, uniq_name = count(1), count()
    clear_directory(save_dir)
    save_dir = os.path.join(save_dir, 'train_files')
    header = ['code', 'x', 'y', 'time']
    drop_fields = ['code']

    for user_path in glob.glob(os.path.join(data_dir, '*')):
        user_name = f"user{next(user_num):02}"
        if verbose >= 2: print(f">> {user_name} ({os.path.basename(user_path)})")
        for session_path in glob.glob(os.path.join(user_path, "RawData/*.txt")):
            # PREPROCESSING
            db = pd.read_csv(session_path, sep=' ', names=header, index_col=False)
            db.drop(drop_fields, axis=1, inplace=True)
            db = db[['time', 'x,' 'y']]
            db.time = (db.time - db.time.iloc[0]) / 1000.0
            db = preprocessing(db, check_size=(verbose >= 3))
            # if check_min_size(db): continue  # all sessions are small

            # PRESERVATION
            train_save_path = os.path.join(save_dir, user_name)
            test_save_path = train_save_path.replace('train', 'test')
            os.makedirs(train_save_path, exist_ok=True)
            os.makedirs(test_save_path, exist_ok=True)
            train_save_path = os.path.join(train_save_path, f"session_{next(uniq_name):05}.csv")
            test_save_path = os.path.join(test_save_path, f"session_{next(uniq_name):05}.csv")

            train_db = db[:int(db.index.size * 0.7)].copy()
            test_db = db[-int(db.index.size * 0.3):].copy()
            test_db.time -= test_db.time.iloc[0]
            train_db.to_csv(train_save_path, index=False)
            test_db.to_csv(test_save_path, index=False)


if __name__ == '__main__':
    start_time = time.time()
    print(f"{COLOR['italics']}CHAOSHEN{COLOR['none']} Run!")
    chaoshen_preprocessing(verbose=3)
    print(f"run time: {time.time() - start_time:.3f}")

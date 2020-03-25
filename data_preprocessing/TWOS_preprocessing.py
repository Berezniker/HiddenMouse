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
#    |-- time        <-- converted to format: sec
#    |-- username    <-- deleted
#    |-- processname <-- deleted
#    |-- message_id  <-- deleted
#    |-- x_pos: int  <-- renamed: 'x'
#    |-- y_pos: int  <-- renamed: 'y'
#    |-- record_info <-- deleted
#    |-- hwnd        <-- deleted
######################################################################


def twos_preprocessing(data_dir: str = '../../original_dataset/TWOS_original/mouse_ano',
                       save_dir: str = '../../dataset/TWOS',
                       verbose: int = 0) -> None:
    uniq_name = count()

    clear_directory(save_dir)
    save_dir = os.path.join(save_dir, 'train_files')

    for user_path in glob.glob(os.path.join(data_dir, 'User*')):
        user_name = os.path.basename(user_path).lower()
        user_name = user_name[:user_name.find('.')]
        if verbose >=2: print(f">> {user_name}")
        if os.path.exists(user_path):
            db = pd.read_csv(user_path, names=['time', 'tmp'])
            db.drop(db[db.tmp.apply(lambda x: "REFRESH" in x)].index, inplace=True)
            db['x'] = db.tmp.str.split(';').str.get(2)
            db['y'] = db.tmp.str.split(';').str.get(3)
            db.dropna(inplace=True)
            db.x = db.x.astype(int)
            db.y = db.y.astype(int)
            db.index = np.arange(len(db))
            db.time = (pd.to_datetime(db.time) - pd.to_datetime(db.time[0].split(' ')[0]))\
                .dt.total_seconds()  # awfully!
            db.time -= db.time.iloc[0]
            db.time += db.tmp.str.split(';').str.get(0).astype(float) / 1000.0
            db.drop(['tmp'], axis=1, inplace=True)
            preprocessing(db)

            train_save_path = os.path.join(save_dir, user_name)
            test_save_path = train_save_path.replace('train', 'test')
            if not os.path.exists(train_save_path):
                os.makedirs(train_save_path)
            if not os.path.exists(test_save_path):
                os.makedirs(test_save_path)
            train_save_path = os.path.join(train_save_path, f"session_{next(uniq_name):02}")
            test_save_path = os.path.join(test_save_path, f"session_{next(uniq_name):02}")

            train_db = db[:int(db.index.size * 0.7)].copy()
            test_db = db[-int(db.index.size * 0.3):].copy()
            test_db.index = np.arange(len(test_db))
            test_db.time -= test_db.time.iloc[0]
            train_db.to_csv(train_save_path, index=False)
            test_db.to_csv(test_save_path, index=False)


if __name__ == '__main__':
    start_time = time.time()
    print(f"{COLOR['italics']}TWOS{COLOR['none']} Run!")
    twos_preprocessing(verbose=2)
    print(f"run time: {time.time() - start_time:.3f}")

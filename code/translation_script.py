from itertools import count
import pandas as pd
import time
import glob
import os

######################################################################
# Kazachuk's dataset
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


if __name__ == '__main__':
    data_dir = '../Local_preprocessed_big'
    save_dir = '../dataset2'
    file_name = 'MOUSE.csv'
    rename_fields = {'x_pos': 'x', 'y_pos': 'y'}
    drop_fields = ['username', 'processname', 'message_id', 'record_info', 'hwnd']
    start_time = time.time()
    uniq_name = count()

    print('Run!')
    for user_path in glob.glob(os.path.join(data_dir, 'User*')):
        user_name = os.path.basename(user_path).lower()
        for intermediate_path in glob.glob(os.path.join(user_path, '*')):
            for session_path in glob.glob(os.path.join(intermediate_path, 'User*')):
                file_path = os.path.join(session_path, file_name)
                if os.path.exists(file_path):
                    db = pd.read_csv(file_path)
                    db.rename(rename_fields, axis=1, inplace=True)
                    db.drop(drop_fields, axis=1, inplace=True)
                    db.time = pd.to_timedelta(pd.to_datetime(db.time)).dt.total_seconds()
                    db.drop(labels=[0], axis=0, inplace=True)
                    db.time -= db.time.iloc[0]
                    for train_test in ['train_files', 'test_files']:
                        save_path = os.path.join(save_dir, train_test)
                        save_path = os.path.join(save_path, user_name)
                        if not os.path.exists(save_path):
                            os.makedirs(save_path)
                        db.to_csv(os.path.join(save_path, f"session_{next(uniq_name):010}"),
                                  index=False)
    print(f"run time: {time.time() - start_time}")

from utils.combine_sessions import combine_sessions
from utils.quntile_encoding import quantile_encoding
from feature_extraction.my_feature import *
from feature_extraction.ars_feature import *
from datetime import datetime
from itertools import chain
import pandas as pd
import time
import glob
import os

TIME_THRESHOLD = 5.0
LOG_FILE_NAME = f"../log_file/log_feature {datetime.today().__str__()[:-7].replace(':', '-')}.txt"
LOG_FILE = None


def printf(*args) -> None:
    if LOG_FILE is not None:
        LOG_FILE.write(*args)
        LOG_FILE.write('\n')
        LOG_FILE.flush()
    print(*args)


def OneHotEncoder(x: np.array,
                  n_classes: int,
                  prefix: str = 'bin') -> pd.DataFrame:
    data = np.eye(n_classes)[x]
    columns = [f"{prefix}_{i}" for i in range(n_classes)]
    return pd.DataFrame(data=data, columns=columns, dtype='uint8')


# TODO need to speed up
def split_dataframe(db: pd.DataFrame,
                    time_threshold: float = TIME_THRESHOLD,
                    min_n_actions: int = 5) -> pd.DataFrame:
    time, max_time = 0, db.time.max()
    while time + time_threshold < max_time:
        one_split = db.loc[(db.time >= time) & (db.time <= time + time_threshold)]
        time = db.loc[db.time > time + time_threshold].time.values[0]
        if one_split.index.size > min_n_actions:
            yield one_split
    if db.loc[db.time >= time].index.size > min_n_actions:
        yield db.loc[db.time >= time]


def extract_features(database: pd.DataFrame,
                     only_one_segment: bool = False,
                     only_one_feature: bool = False) -> pd.DataFrame:
    extraction_function = list([
        direction_bin, actual_distance, actual_distance_bin,
        curve_length, curve_length_bin, length_ratio, actual_speed, curve_speed,
        curve_acceleration, mean_movement_offset, mean_movement_error,
        mean_curvature, mean_curvature_change_rate, mean_curvature_velocity,
        mean_curvature_velocity_change_rate, mean_angular_velocity
    ])  # len(...) = 16 --(one-hot-encoder)--> 61
    ars_extracrion_function = list([
        min_x, max_x, mean_x, std_x, max_min_x,
        min_y, max_y, mean_y, std_y, max_min_y,
        min_vx, max_vx, mean_vx, std_vx, max_min_vx,
        min_vy, max_vy, mean_vy, std_vy, max_min_vy,
        min_v, max_v, mean_v, std_v, max_min_v,
        min_a, max_a, mean_a, std_a, max_min_a,
        min_j, max_j, mean_j, std_j, max_min_j,
        min_am, max_am, mean_am, std_am, max_min_am,
        min_av, max_av, mean_av, std_av, max_min_av,
        min_c, max_c, mean_c, std_c, max_min_c,
        min_ccr, max_ccr, mean_ccr, std_ccr, max_min_ccr,
        duration_of_movement, straightness,
        TCM, SC, M3, M4, TCrv, VCrv
    ])  # len(...) = 63
    features = dict()
    for segment in split_dataframe(database):
        for extractor in chain(extraction_function, ars_extracrion_function):
            features.setdefault(extractor.__name__, []).append(extractor(segment))
            if only_one_feature:
                printf('>>>> [!] only_one_feature')
                break
        if only_one_segment:
            printf('>>> [!] only_one_segment')
            break

    return pd.DataFrame(features)


def run(data_path: str,
        save_path: str = None,
        save_features: bool = False,
        only_one_user: bool = False,
        only_one_session: bool = False,
        start_from: int = 0) -> None:
    if save_features:
        if save_path is None:
            save_path = data_path.replace('files', 'features')
    else:
        printf("[Warning] Data will not be saved (save_features=False)")
    printf(f'\n{data_path} --> {save_path}\n')

    for user in glob.glob(os.path.join(data_path, 'user*')):
        user_start_time = time.time()
        user_name = os.path.basename(user)
        printf(f'> {user_name}')
        if int(user_name[4:]) < start_from:
            continue
        n_session = len(list(os.listdir(user)))
        for i, session in enumerate(glob.glob(os.path.join(user, 'session*')), 1):
            session_start_time = time.time()
            session_name = os.path.basename(session)
            # vvvvvv
            features = extract_features(pd.read_csv(session))
            """
            # One-Hot Encoding:
            features = features.join(OneHotEncoder(features.pop('direction_bin').values,
                                                   n_classes=8, prefix='bin'))
            features = features.join(OneHotEncoder(features.pop('actual_distance_bin').values,
                                                   n_classes=20, prefix='ad_bin'))
            features = features.join(OneHotEncoder(features.pop('curve_length_bin').values,
                                                   n_classes=20, prefix='cl_bin'))
            """
            # ^^^^^^
            if save_features:
                session_dir = os.path.join(save_path, user_name)
                if not os.path.exists(session_dir):
                    os.makedirs(session_dir)
                features.to_csv(os.path.join(session_dir, session_name), index=False)
            printf(f'>> [{i:03}/{n_session}] {session_name} time: {time.time() - session_start_time:6.3f} sec')
            if only_one_session:
                printf('>> [!] only_one_session')
                break
        printf(f'> {user_name} time: {time.time() - user_start_time:.3f} sec\n')
        if only_one_user:
            printf('> [!] only_one_user')
            break


if __name__ == '__main__':
    if input("Keep a log? Y/N: ").upper() == 'Y':
        print("log [ON]")
        LOG_FILE = open(LOG_FILE_NAME, mode='w')
    else:
        print("log [OFF]")

    start_time = time.time()
    printf("train: RUN!")
    train_start_time = time.time()
    # run(data_path="../../dataset/BALABIT/train_files", save_features=True)
    run(data_path="../../dataset/DATAIIT/train_files", save_features=True)
    # run(data_path="../../dataset/TWOS/train_files", save_features=True)
    printf(f"train_time: {(time.time() - train_start_time) / 60.0:.1f} min\n\n")

    printf("test: RUN!")
    test_start_time = time.time()
    # run(data_path="../../dataset/BALABIT/test_files", save_features=True)
    run(data_path="../../dataset/DATAIIT/test_files", save_features=True)
    # run(data_path="../../dataset/TWOS/test_files", save_features=True)
    printf(f"test_time: {(time.time() - test_start_time) / 60.0:.1f} min\n")

    # combine_sessions(verbose=3)
    # quantile_encoding(verbose=3)

    printf(f"main_time: {(time.time() - start_time) / 60.0:.1f} min\n")

    if LOG_FILE is not None:
        LOG_FILE.close()

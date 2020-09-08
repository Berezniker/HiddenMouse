from utils.combine_sessions import combine_sessions
from utils.quantile_encoding import quantile_encoding
from feature_extraction.my_feature import *
from feature_extraction.ars_feature import *
from datetime import datetime
from itertools import chain
import pandas as pd
import time
import glob
import os

TIME_THRESHOLD = 5.0
ACTION_THRESHOLD = 5
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


# TODO need to speed up !!!
def split_dataframe(db: pd.DataFrame,
                    time_threshold: float = TIME_THRESHOLD,
                    min_n_actions: int = ACTION_THRESHOLD) -> pd.DataFrame:
    ctime, max_time = 0, db.time.max()
    db.reset_index(drop=True)
    while ctime + time_threshold < max_time:
        one_split = db.loc[(db.time >= ctime) & (db.time < ctime + time_threshold), :]
        if one_split.index.size > min_n_actions:
            yield one_split
        idx = one_split.index[-1] + 1
        ctime = db.loc[idx, 'time'] if idx < db.index.size else max_time
    if db.loc[(db.time >= ctime), :].index.size > min_n_actions:
        yield db.loc[(db.time >= ctime), :]


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
    ars_extracrion_function = list()  # TODO
    features = dict()

    for segment in split_dataframe(database):
        for extractor in chain(extraction_function, ars_extracrion_function):
            # ------------------------------------------------------------------ #
            features.setdefault(extractor.__name__, []).append(extractor(segment))
            # ------------------------------------------------------------------ #
            if only_one_feature:
                printf('>>>> [!] only_one_feature')
                break
        if only_one_segment:
            printf('>>> [!] only_one_segment')
            break

    return pd.DataFrame(features).round(decimals=5)


def run(data_path: str,
        save_path: str = None,
        save_features: bool = False,
        only_one_user: bool = False,
        only_one_session: bool = False) -> None:
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
        n_session = len(list(os.listdir(user)))
        for i, session in enumerate(glob.glob(os.path.join(user, 'session*')), 1):
            session_start_time = time.time()
            session_name = os.path.basename(session)
            # --------------------------- #
            db = pd.read_csv(session)
            features = extract_features(db)
            # --------------------------- #
            """
            # One-Hot Encoding:
            features = features.join(OneHotEncoder(features.pop('direction_bin').values,
                                                   n_classes=8, prefix='bin'))
            features = features.join(OneHotEncoder(features.pop('actual_distance_bin').values,
                                                   n_classes=20, prefix='ad_bin'))
            features = features.join(OneHotEncoder(features.pop('curve_length_bin').values,
                                                   n_classes=20, prefix='cl_bin'))
            """
            if save_features:
                session_dir = os.path.join(save_path, user_name)
                os.makedirs(session_dir, exist_ok=True)
                features.to_csv(os.path.join(session_dir, session_name), index=False)
            printf(f'>> [{i:03}/{n_session}] {session_name} '
                   f' data.size = {db.index.size:7} '
                   f' features.size = {features.index.size:4} '
                   f' time: {time.time() - session_start_time:5.1f} sec')
            if only_one_session:
                printf('>> [!] only_one_session')
                break
        printf(f'> {user_name} time: {(time.time() - user_start_time) / 60.0:.1f} min\n')
        if only_one_user:
            printf('> [!] only_one_user')
            break


if __name__ == '__main__':
    start_time = time.time()
    for dataset in ["BALABIT", "DATAIIT", "TWOS"]:  # TODO
        log_file_path = f"../../dataset/{dataset}/log_{dataset}_feature_extraction.txt"
        LOG_FILE = open(log_file_path, mode='w')
        LOG_FILE.write(f"{datetime.now().isoformat(sep=' ')[:-7]}\n\n")
        dataset_start_time = time.time()
        printf(f"{dataset}")
        for type_file in ['test']:
            type_start_time = time.time()
            printf(f"{type_file}")
            # --------------------------------------------------------------------------- #
            run(data_path=f"../../dataset/{dataset}/{type_file}_files", save_features=True)
            # --------------------------------------------------------------------------- #
            printf(f"{type_file}_time: {(time.time() - type_start_time) / 60.0:.1f} min\n\n")
        printf(f"{dataset}_time: {(time.time() - dataset_start_time) / 60.0:.1f} min\n\n")
        LOG_FILE.close()
    print(f"main_time: {(time.time() - start_time) / 60.0:.1f} min\n")

    combine_sessions(verbose=3)
    # quantile_encoding(verbose=3)

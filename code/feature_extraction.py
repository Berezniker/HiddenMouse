import pandas as pd
import numpy as np
import time as t
import glob
import os


######################################################################
# Balabit dataset
#    |
#    |-- record timestamp (in sec): float  <-- deleted
#    |-- client timestamp (in sec): float  <-- renamed: 'time'
#    |-- button: ['NoButton', 'Left', 'Scroll', 'Right']
#    |                                ~~~~~~~~ <-- deleted
#    |-- state: ['Move', 'Pressed', 'Released', 'Drag', 'Down', 'Up']
#    |                                      deleted --> ~~~~~~  ~~~~
#    |-- x: int
#    |-- y: int
######################################################################

# feature extraction function template:
#
# def feature_name(db):
#     """
#     extract features <features_name> from the database <db>
#     :param db: current session database
#     :return: feature value
#     """

EPS = 1e-5
LOG_FILE_NAME = 'log_feature.txt'
LOG_FILE = None


def printf(*args, to_file=False, to_display=True):
    global LOG_FILE
    if to_file:
        LOG_FILE.write(*args)
        LOG_FILE.write('\n')
    if to_display:
        print(*args)
    if not (to_file or to_display):
        print('* silence *', end='')


def get_bin(dist, threshold=1000):
    if dist < 0:
        return -1
    elif dist <= threshold:
        return dist % (threshold // 20)
    elif dist <= 2 * threshold:
        return 20 + dist % (threshold // 10)
    elif dist <= 3 * threshold:
        return 30 + dist % (threshold // 5)
    elif dist <= 4 * threshold:
        return 35 + dist % (threshold // 2)
    else:
        return 38


def get_grad(val):
    return np.array([val[1], *(val[2:] - val[:-2]), -val[-2]])


def get_det(db):
    Pn_Po = np.array([db.x.iloc[-1] - db.x.iloc[0], db.y.iloc[-1] - db.y.iloc[0]])
    x0, y0 = db.x.iloc[0], db.y.iloc[0]
    det = np.array([np.linalg.det([Pn_Po, [x - x0, y - y0]])
                    for x, y in zip(db.x.iloc[1:].values, db.y.iloc[1:].values)])
    return det


def direction_bin(db):
    grad_x, grad_y = get_grad(db.x.values), get_grad(db.y.values)
    direction = np.rad2deg(np.arctan2(grad_y, grad_x)) + 180
    direction = (direction % 8).round().astype(np.uint8)
    return np.argmax(np.bincount(direction))


def actual_distance(db):
    return ((db.x.iloc[0] - db.x.iloc[-1]) ** 2 +
            (db.y.iloc[0] - db.y.iloc[-1]) ** 2) ** 0.5


def actual_distance_bin(db, threshold=1000):
    return get_bin(actual_distance(db), threshold=threshold)


def curve_length(db):
    return (((db.x.iloc[:-1].values - db.x.iloc[1:].values) ** 2 +
             (db.y.iloc[:-1].values - db.y.iloc[1:].values) ** 2) ** 0.5).sum()


def curve_length_bin(db, threshold=1000):
    return get_bin(curve_length(db), threshold=threshold)


def length_ratio(db):
    return curve_length(db) / (actual_distance(db) + EPS)


def actual_speed(db):
    return actual_distance(db) / (db.time.iloc[-1] - db.time.iloc[0] + EPS)


def curve_speed(db):
    return ((((db.x.iloc[:-1].values - db.x.iloc[1:].values) ** 2 +
              (db.y.iloc[:-1].values - db.y.iloc[1:].values) ** 2) ** 0.5)
            / (db.time.iloc[1:].values - db.time.iloc[:-1].values + EPS)
            ).mean()


def curve_acceleration(db):
    return curve_speed(db) / (db.time.iloc[-1] - db.time.iloc[0] + EPS)


def mean_movement_offset(db):
    Pn_Po = np.array([db.x.iloc[-1] - db.x.iloc[0], db.y.iloc[-1] - db.y.iloc[0]])
    return (get_det(db) / (np.linalg.norm(Pn_Po) + EPS)).mean()


def mean_movement_error(db):
    Pn_Po = np.array([db.x.iloc[-1] - db.x.iloc[0], db.y.iloc[-1] - db.y.iloc[0]])
    return (abs(get_det(db)) / (np.linalg.norm(Pn_Po) + EPS)).mean()


def mean_movement_variability(db):
    return ((db.y.iloc[1:-1] - mean_movement_offset(db)) ** 2).mean() ** 0.5


def mean_curvature(db):
    return (np.arctan2(db.y, db.x) / (db.x ** 2 + db.y ** 2) ** 0.5).mean()


def mean_curvature_change_rate(db):
    return (np.arctan2(db.y.iloc[:-1].values, db.x.iloc[:-1].values) /
            (((db.x.iloc[-1] - db.x.iloc[:-1].values) ** 2 +
              (db.y.iloc[-1] - db.y.iloc[:-1].values) ** 2) ** 0.5 + EPS)).mean()


def mean_curvature_velocity(db):
    return mean_curvature(db) / (db.time.iloc[-1] - db.time.iloc[0] + EPS)


def mean_curvature_velocity_change_rate(db):
    return mean_curvature(db) / (db.time.iloc[-1] - db.time.iloc[0] + EPS) ** 2


def mean_angular_velocity(db):
    a = np.array([db.x.iloc[:-2].values - db.x.iloc[1:-1].values,
                  db.y.iloc[:-2].values - db.y.iloc[1:-1].values])
    b = np.array([db.x.iloc[2:].values - db.x.iloc[1:-1].values,
                  db.y.iloc[2:].values - db.y.iloc[1:-1].values])
    angle = np.arccos(np.sum(a * b, axis=0) /
                      (np.linalg.norm(a, axis=0) * np.linalg.norm(b, axis=0) + EPS))
    return (angle / (db.time.iloc[2:].values - db.time.iloc[:-2].values + EPS)).mean()


def database_preprocessing(db):
    db.rename({'client timestamp': 'time'}, axis=1, inplace=True)
    db.drop('record timestamp', axis=1, inplace=True)
    db.drop_duplicates(inplace=True)
    db.drop_duplicates(subset='time', inplace=True)
    db.drop(db[db.button == 'Scroll'].index, inplace=True)
    db.drop(db[db.x > 2000].index, inplace=True)
    db.drop(db[db.y > 1200].index, inplace=True)


# TODO need to speed up
def split_dataframe(db, time_threshold=1, min_n_actions=2):
    time, max_time = 0, db.time.max()
    while time + time_threshold < max_time:
        one_split = db.loc[(db.time >= time) & (db.time <= time + time_threshold)]
        time = db.loc[db.time > time + time_threshold].time.values[0]
        if one_split.index.size > min_n_actions:
            yield one_split
    if db.loc[db.time >= time].index.size > 2:
        yield db.loc[db.time >= time]


def extract_features(database, only_one_segment=False, only_one_feature=False):
    database_preprocessing(database)
    extraction_function = [
        direction_bin, actual_distance, actual_distance_bin,
        curve_length, curve_length_bin, length_ratio, actual_speed, curve_speed,
        curve_acceleration, mean_movement_offset, mean_movement_error,
        mean_curvature, mean_curvature_change_rate, mean_curvature_velocity,
        mean_curvature_velocity_change_rate, mean_angular_velocity
    ]
    features = dict()
    for i, segment in enumerate(split_dataframe(database), 1):
        for extractor in extraction_function:
            features.setdefault(extractor.__name__, []).append(extractor(segment))
            if only_one_feature:
                printf('>>>> [!] only_one_feature')
                break
        if only_one_segment or only_one_feature:
            printf('>>> [!] only_one_segment')
            break
    return features


def run(mode, only_one_user=False, only_one_session=False):
    data_path = f'../dataset/{mode}_files'
    save_path = f'../features/{mode}_features'

    for user in glob.glob(os.path.join(data_path, 'user*')):
        printf(f'> {os.path.basename(user)}')
        user_start_time = t.time()
        n_session = len(list(os.listdir(user)))
        for i, session in enumerate(glob.glob(os.path.join(user, 'session*')), 1):
            session_start_time = t.time()
            database = pd.read_csv(session)
            # vvvvvvvvvvv
            features_dict = extract_features(database)
            # ^^^^^^^^^^^
            features = pd.DataFrame(features_dict)
            session_dir = os.path.join(save_path, os.path.basename(user))
            if not os.path.exists(session_dir):
                os.makedirs(session_dir)
            features.to_csv(os.path.join(session_dir, os.path.basename(session)),
                            index=False)
            printf(f'>> [{i:03}/{n_session}] {os.path.basename(session)} time: {t.time() - session_start_time:.3f} sec')
            if only_one_session:
                printf('>> [!] only_one_session')
                break
        printf(f'> {os.path.basename(user)} time: {t.time() - user_start_time:.3f} sec')
        if only_one_user or only_one_session:
            printf('> [!] only_one_user')
            break


if __name__ == '__main__':
    # LOG_FILE = open(LOG_FILE_NAME, mode='a')
    start_time = t.time()
    printf('train: RUN!')
    train_start_time = t.time()
    run('train')
    printf(f'train_time: {t.time() - train_start_time:.3f} sec')
    printf('\n')
    printf('test: RUN!')
    test_start_time = t.time()
    run('test')
    printf(f'test_time: {t.time() - test_start_time:.3f} sec')
    printf('\n')
    printf(f'main_time: {t.time() - start_time:.3f} sec')
    # LOG_FILE.close()

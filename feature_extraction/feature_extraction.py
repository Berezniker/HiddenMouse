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
#    |-- state: ['Move', 'Pressed', 'Released', 'Drag', 'Down', 'Up']
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


def get_bin(dist, threshold=1000):
    if dist < 0:
        return -1
    elif dist <= threshold:
        return dist % 50
    elif dist <= 2 * threshold:
        return 20 + dist % 100
    elif dist <= 3 * threshold:
        return 30 + dist % 200
    elif dist <= 4 * threshold:
        return 35 + dist % 500
    else:
        return 38


def direction_bin(db):
    pass  # TODO


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


def mean_movement_offset(db):  # TODO
    Pn_Po = np.array([db.x.iloc[-1] - db.x.iloc[0], db.y.iloc[-1] - db.y.iloc[0]])
    return (np.linalg.det(np.array([Pn_Po, [db.x.iloc[1:].values - db.x.iloc[0], db.y.iloc[1:].values - db.y.iloc[0]]]))
            / np.linalg.norm(Pn_Po)).mean()


def mean_movement_error(db):
    Pn_Po = np.array([db.x.iloc[-1] - db.x.iloc[0], db.y.iloc[-1] - db.y.iloc[0]])
    pass  # TODO


def mean_movement_variability(db):
    movement_offset = mean_movement_offset(db)
    return ((db.y.iloc[1:-1] - movement_offset) ** 2).mean() ** 0.5


def mean_curvature(db):
    return (np.arctan(db.y / db.x) / (db.x ** 2 + db.y ** 2) ** 0.5).mean()


def mean_curvature_change_rate(db):
    return (np.arctan(db.y / db.x) /
            ((db.x.iloc[-1] - db.x.iloc[:-1]) ** 2 + (db.y.iloc[-1] - db.y.iloc[:-1]) ** 2) ** 0.5).mean()


def mean_curvature_velocity(db):
    return mean_curvature(db) / (db.time.iloc[-1] - db.time.iloc[0] + EPS)


def mean_curvature_velocity_change_rate(db):
    return mean_curvature(db) / (db.time.iloc[-1] - db.time.iloc[0] + EPS) ** 2


def mean_angular_velocity(db):
    pass  # TODO


def database_preprocessing(db):
    db.rename({'client timestamp': 'time'}, axis=1, inplace=True)
    db.drop('record timestamp', axis=1, inplace=True)
    db.drop_duplicates(inplace=True)
    db.drop_duplicates(subset='time', inplace=True)
    db.drop(db[db.x > 2000].index, inplace=True)
    db.drop(db[db.y > 1200].index, inplace=True)
    # db.loc[db.time.duplicated(), 'time'] += 5e-4


# TODO need to speed up
def split_dataframe(db, time_threshold=1.0):
    time, max_time = 0, db['time'].max()
    while time + time_threshold < max_time:
        yield db.loc[(db['time'] >= time) & (db['time'] <= time + time_threshold)]
        time = db.loc[db['time'] > time + time_threshold].time.values[0]
    yield db.loc[db['time'] >= time]


def extract_features(database):
    database_preprocessing(database)
    # don't forget to add your function here:
    extraction_function = [
        direction_bin, actual_distance, actual_distance_bin,
        curve_length, curve_length_bin, length_ratio, actual_speed, curve_speed,
        curve_acceleration,
        mean_curvature, mean_curvature_change_rate, mean_curvature_velocity,
        mean_curvature_velocity_change_rate,
    ]
    features = dict()
    for segment in split_dataframe(database):
        for extractor in extraction_function:
            features.setdefault(extractor.__name__, []).append(extractor(segment))
        break
    return features


def run(mode):
    data_path = f'../dataset/{mode}_files'
    save_path = f'../features/{mode}_features'

    for user in glob.glob(os.path.join(data_path, 'user*')):
        for session in glob.glob(os.path.join(user, 'session*')):
            database = pd.read_csv(session)
            # vvvvvvvvvv
            features_dict = extract_features(database)
            # ^^^^^^^^^^
            features = pd.DataFrame(features_dict)
            session_dir = os.path.join(save_path, os.path.basename(user))
            if not os.path.exists(session_dir):
                os.makedirs(session_dir)
            features.to_csv(os.path.join(session_dir, os.path.basename(session)),
                            index=False)
            break
        break


if __name__ == '__main__':
    start_time = t.time()
    run('train')
    print('train_time:', t.time() - start_time)
    # start_time = t.time()
    # run('test')
    # print('train_time:', t.time() - start_time)

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


def direction_bin(db):
    pass  # TODO


def actual_distance(db):
    return ((db.x.iloc[0] - db.x.iloc[-1]) ** 2 +
            (db.y.iloc[0] - db.y.iloc[-1]) ** 2) ** 0.5


def actual_distance_bin(db):
    pass  # TODO


def curve_length(db):
    return (((db.x.iloc[:-1].values - db.x.iloc[1:].values) ** 2 +
             (db.y.iloc[:-1].values - db.y.iloc[1:].values) ** 2) ** 0.5).sum()


def curve_length_bin(db):
    pass  # TODO


def length_ratio(db):
    return curve_length(db) / (actual_distance(db) + 1e-5)


def actual_speed(db):
    return actual_distance(db) / (db.time.iloc[-1] - db.time.iloc[0])


def curve_speed(db):
    return ((((db.x.iloc[:-1].values - db.x.iloc[1:].values) ** 2 +
              (db.y.iloc[:-1].values - db.y.iloc[1:].values) ** 2) ** 0.5)
            / (db.time.iloc[1:].values - db.time.iloc[:-1].values)
            ).mean()


def curve_acceleration(db):
    return curve_speed(db) / (db.time.iloc[-1] - db.time.iloc[0])


def database_preprocessing(db):
    db.rename({'client timestamp': 'time'}, axis=1, inplace=True)
    db.drop('record timestamp', axis=1, inplace=True)
    db.drop_duplicates(inplace=True)
    db.drop_duplicates(subset='time', inplace=True)
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
    extraction_function = [actual_distance, curve_length, length_ratio,
                           actual_speed, curve_speed, curve_acceleration]
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

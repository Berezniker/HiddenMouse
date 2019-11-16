import pandas as pd
import numpy as np
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


def database_preprocessing(db):
    db.rename({'client timestamp': 'time'}, axis=1, inplace=True)
    db.drop('record timestamp', axis=1, inplace=True)
    db.drop_duplicates(inplace=True)


def split_dataframe(db, time_threshold=1.0):
    time, max_time = 0, db['time'].max()
    while time + time_threshold < max_time:
        yield db.loc[(db['time'] >= time) & (db['time'] <= time + time_threshold)]
        time = db.loc[db['time'] > time + time_threshold].time.values[0]
    yield db.loc[db['time'] >= time]


def extract_features(database):
    database_preprocessing(database)
    features = dict()
    for segment in split_dataframe(database):
        features.setdefault('actual_distance', []).append(actual_distance(segment))
        features.setdefault('curve_length', []).append(curve_length(segment))
        # TODO
        break
    return features


def extractor(mode):
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
    extractor('train')
    # extractor('test')

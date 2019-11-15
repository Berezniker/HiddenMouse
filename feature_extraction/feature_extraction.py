import pandas as pd
import glob
import os


######################################################################
# Balabit dataset
#    |
#    |-- record timestamp (in sec): float
#    |-- client timestamp (in sec): float
#    |-- button: ['NoButton', 'Left', 'Scroll', 'Right']
#    |-- state: ['Move', 'Pressed', 'Released', 'Drag', 'Down', 'Up']
#    |-- x: int
#    |-- y: int
######################################################################

# feature extraction function template:
#
# def feature_name(db, df):
#     """
#     extract the features <features_name>
#     :param db: current session database
#     :param df: features of the current session
#     :return: None
#     """
#     ...
#     df['feature_name'] = ...
#     ...


def actual_distance(db, df):
    df['actual_distance'] = (db.x ** 2 + db.y ** 2) ** 0.5


def actual_distance_bin(db, df):
    pass  # TODO


def extract_features(database, features):
    actual_distance(database, features)
    # TODO


def extractor(mode):
    data_path = f'../dataset/{mode}_files'
    save_path = f'../features/{mode}_features'

    for user in glob.glob(os.path.join(data_path, 'user*')):
        for session in glob.glob(os.path.join(user, 'session*')):
            database = pd.read_csv(session)
            features = pd.DataFrame()
            # vvvvvvvvvv
            extract_features(database, features)
            # ^^^^^^^^^^
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

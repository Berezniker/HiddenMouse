import pandas as pd
import numpy as np

EPS = 1e-5

# feature extraction function template:
#
# def feature_name(db):
#     """
#     extract features <features_name> from the database <db>
#     :param db: current session database
#     :return: feature value
#     """

def get_bin(dist: int, threshold: float = 1000) -> int:
    if dist < 0:
        return -1
    elif dist <= threshold:
        return dist % 5
    elif dist <= 2 * threshold:
        return 5 + dist % 10
    elif dist <= 3 * threshold:
        return 15 + dist % 4
    else:
        return 19


def get_grad(val: np.array) -> np.array:
    return np.array([val[1], *(val[2:] - val[:-2]), -val[-2]])


def get_det(db: pd.DataFrame) -> np.array:
    Pn_Po = np.array([db.x.iloc[-1] - db.x.iloc[0], db.y.iloc[-1] - db.y.iloc[0]])
    x0, y0 = db.x.iloc[0], db.y.iloc[0]
    det = np.array([np.linalg.det([Pn_Po, [x - x0, y - y0]])
                    for x, y in zip(db.x.iloc[1:].values, db.y.iloc[1:].values)])
    return det


def direction_bin(db: pd.DataFrame, n_bin: int = 8) -> np.array:
    grad_x, grad_y = get_grad(db.x.values), get_grad(db.y.values)
    direction = np.rad2deg(np.arctan2(grad_y, grad_x)) + 180
    direction = (direction % n_bin).astype(np.uint8)
    return np.argmax(np.bincount(direction))


def actual_distance(db: pd.DataFrame) -> float:
    return ((db.x.iloc[0] - db.x.iloc[-1]) ** 2 +
            (db.y.iloc[0] - db.y.iloc[-1]) ** 2) ** 0.5


def actual_distance_bin(db: pd.DataFrame, threshold: float = 1000) -> int:
    return get_bin(int(actual_distance(db)), threshold=threshold)


def curve_length(db: pd.DataFrame) -> float:
    return (((db.x.iloc[:-1].values - db.x.iloc[1:].values) ** 2 +
             (db.y.iloc[:-1].values - db.y.iloc[1:].values) ** 2) ** 0.5).sum()


def curve_length_bin(db: pd.DataFrame, threshold: float = 1000) -> int:
    return get_bin(int(curve_length(db)), threshold=threshold)


def length_ratio(db: pd.DataFrame) -> float:
    return curve_length(db) / (actual_distance(db) + EPS)


def actual_speed(db: pd.DataFrame) -> float:
    return actual_distance(db) / (db.time.iloc[-1] - db.time.iloc[0] + EPS)


def curve_speed(db: pd.DataFrame) -> float:
    return ((((db.x.iloc[:-1].values - db.x.iloc[1:].values) ** 2 +
              (db.y.iloc[:-1].values - db.y.iloc[1:].values) ** 2) ** 0.5)
            / (db.time.iloc[1:].values - db.time.iloc[:-1].values + EPS)
            ).mean()


def curve_acceleration(db: pd.DataFrame) -> float:
    return curve_speed(db) / (db.time.iloc[-1] - db.time.iloc[0] + EPS)


def mean_movement_offset(db: pd.DataFrame) -> float:
    Pn_Po = np.array([db.x.iloc[-1] - db.x.iloc[0], db.y.iloc[-1] - db.y.iloc[0]])
    return (get_det(db) / (np.linalg.norm(Pn_Po) + EPS)).mean()


def mean_movement_error(db: pd.DataFrame) -> float:
    Pn_Po = np.array([db.x.iloc[-1] - db.x.iloc[0], db.y.iloc[-1] - db.y.iloc[0]])
    return (abs(get_det(db)) / (np.linalg.norm(Pn_Po) + EPS)).mean()


def mean_movement_variability(db: pd.DataFrame) -> float:
    return ((db.y.iloc[1:-1] - mean_movement_offset(db)) ** 2).mean() ** 0.5


def mean_curvature(db: pd.DataFrame) -> float:
    return (np.arctan2(db.y, db.x) / (db.x ** 2 + db.y ** 2) ** 0.5).mean() * 1000


def mean_curvature_change_rate(db: pd.DataFrame) -> float:
    return (np.arctan2(db.y.iloc[:-1].values, db.x.iloc[:-1].values) /
            (((db.x.iloc[-1] - db.x.iloc[:-1].values) ** 2 +
              (db.y.iloc[-1] - db.y.iloc[:-1].values) ** 2) ** 0.5 + EPS)).mean() * 100


def mean_curvature_velocity(db: pd.DataFrame) -> float:
    return mean_curvature(db) / (db.time.iloc[-1] - db.time.iloc[0] + EPS)


def mean_curvature_velocity_change_rate(db: pd.DataFrame) -> float:
    return mean_curvature(db) / (db.time.iloc[-1] - db.time.iloc[0] + EPS) ** 2


def mean_angular_velocity(db: pd.DataFrame) -> float:
    a = np.array([db.x.iloc[:-2].values - db.x.iloc[1:-1].values,
                  db.y.iloc[:-2].values - db.y.iloc[1:-1].values])
    b = np.array([db.x.iloc[2:].values - db.x.iloc[1:-1].values,
                  db.y.iloc[2:].values - db.y.iloc[1:-1].values])
    angle = np.arccos(np.sum(a * b, axis=0) /
                      (np.linalg.norm(a, axis=0) * np.linalg.norm(b, axis=0) + EPS))
    return (angle / (db.time.iloc[2:].values - db.time.iloc[:-2].values + EPS)).mean()

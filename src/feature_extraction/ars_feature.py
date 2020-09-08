import pandas as pd
import numpy as np

EPS = 1e-7

# feature extraction function template:
#
# def feature_name(db):
#     """
#     extract features <features_name> from the database <db>
#     :param db: current session database
#     :return: feature value
#     """

def min_x(db: pd.DataFrame) -> float:
    return db.x.min()

def max_x(db: pd.DataFrame) -> float:
    return db.x.max()

def mean_x(db: pd.DataFrame) -> float:
    return db.x.mean()

def std_x(db: pd.DataFrame) -> float:
    return db.x.std()

def max_min_x(db: pd.DataFrame) -> float:
    return max_x(db) - min_x(db)


def min_y(db: pd.DataFrame) -> float:
    return db.y.min()

def max_y(db: pd.DataFrame) -> float:
    return db.y.max()

def mean_y(db: pd.DataFrame) -> float:
    return db.y.mean()

def std_y(db: pd.DataFrame) -> float:
    return db.y.std()

def max_min_y(db: pd.DataFrame) -> float:
    return max_y(db) - min_y(db)


def velocity_x(db: pd.DataFrame) -> np.ndarray:
    return db.x.diff().values[1:] / (db.time.diff().values[1:] + EPS)

def min_vx(db: pd.DataFrame) -> float:
    return velocity_x(db).min()

def max_vx(db: pd.DataFrame) -> float:
    return velocity_x(db).max()

def mean_vx(db: pd.DataFrame) -> float:
    return velocity_x(db).mean()

def std_vx(db: pd.DataFrame) -> float:
    return velocity_x(db).std()

def max_min_vx(db: pd.DataFrame) -> float:
    return max_vx(db) - min_vx(db)


def velocity_y(db: pd.DataFrame) -> np.ndarray:
    return db.y.diff().values[1:] / (db.time.diff().values[1:] + EPS)

def min_vy(db: pd.DataFrame) -> float:
    return velocity_y(db).min()

def max_vy(db: pd.DataFrame) -> float:
    return velocity_y(db).max()

def mean_vy(db: pd.DataFrame) -> float:
    return velocity_y(db).mean()

def std_vy(db: pd.DataFrame) -> float:
    return velocity_y(db).std()

def max_min_vy(db: pd.DataFrame) -> float:
    return max_vy(db) - min_vy(db)


def velocity(db: pd.DataFrame) -> np.ndarray:
    return np.hypot(velocity_x(db), velocity_y(db))
    # return (velocity_x(db) ** 2 + velocity_y(db) ** 2) ** 0.5

def min_v(db: pd.DataFrame) -> float:
    return velocity(db).min()

def max_v(db: pd.DataFrame) -> float:
    return velocity(db).max()

def mean_v(db: pd.DataFrame) -> float:
    return velocity(db).mean()

def std_v(db: pd.DataFrame) -> float:
    return velocity(db).std()

def max_min_v(db: pd.DataFrame) -> float:
    return max_v(db) - min_v(db)


def acceleration(db: pd.DataFrame) -> np.ndarray:
    return velocity(db) / (db.time.diff().values[1:] + EPS)

def acceleration_x(db: pd.DataFrame) -> np.ndarray:
    return velocity_x(db) / (db.time.diff().values[1:] + EPS)

def acceleration_y(db: pd.DataFrame) -> np.ndarray:
    return velocity_y(db) / (db.time.diff().values[1:] + EPS)

def min_a(db: pd.DataFrame) -> float:
    return acceleration(db).min()

def max_a(db: pd.DataFrame) -> float:
    return acceleration(db).max()

def mean_a(db: pd.DataFrame) -> float:
    return acceleration(db).mean()

def std_a(db: pd.DataFrame) -> float:
    return acceleration(db).std()

def max_min_a(db: pd.DataFrame) -> float:
    return max_a(db) - min_a(db)


def jerk(db: pd.DataFrame) -> np.ndarray:
    return acceleration(db) / (db.time.diff().values[1:] + EPS)

def min_j(db: pd.DataFrame) -> float:
    return jerk(db).min()

def max_j(db: pd.DataFrame) -> float:
    return jerk(db).max()

def mean_j(db: pd.DataFrame) -> float:
    return jerk(db).mean()

def std_j(db: pd.DataFrame) -> float:
    return jerk(db).std()

def max_min_j(db: pd.DataFrame) -> float:
    return max_j(db) - min_j(db)


def angle_of_movement(db: pd.DataFrame) -> np.ndarray:
    return np.arctan2(db.y, db.x)

def min_am(db: pd.DataFrame) -> float:
    return angle_of_movement(db).min()

def max_am(db: pd.DataFrame) -> float:
    return angle_of_movement(db).max()

def mean_am(db: pd.DataFrame) -> float:
    return angle_of_movement(db).mean()

def std_am(db: pd.DataFrame) -> float:
    return angle_of_movement(db).std()

def max_min_am(db: pd.DataFrame) -> float:
    return max_c(db) - min_c(db)


def curvature(db: pd.DataFrame) -> np.ndarray:
    return angle_of_movement(db) / np.hypot(db.x, db.y)

def min_c(db: pd.DataFrame) -> float:
    return curvature(db).min()

def max_c(db: pd.DataFrame) -> float:
    return curvature(db).max()

def mean_c(db: pd.DataFrame) -> float:
    return curvature(db).mean()

def std_c(db: pd.DataFrame) -> float:
    return curvature(db).std()

def max_min_c(db: pd.DataFrame) -> float:
    return max_c(db) - min_c(db)


def curvature_change_rate(db: pd.DataFrame) -> np.ndarray:
    return curvature(db) / np.hypot(db.x, db.y)

def min_ccr(db: pd.DataFrame) -> float:
    return curvature_change_rate(db).min()

def max_ccr(db: pd.DataFrame) -> float:
    return curvature_change_rate(db).max()

def mean_ccr(db: pd.DataFrame) -> float:
    return curvature_change_rate(db).mean()

def std_ccr(db: pd.DataFrame) -> float:
    return curvature_change_rate(db).std()

def max_min_ccr(db: pd.DataFrame) -> float:
    return max_ccr(db) - min_ccr(db)


def angular_velocity(db: pd.DataFrame) -> np.ndarray:
    dt = db.time.diff().values[1:] + EPS
    return np.arctan2(db.y.iloc[1:], db.x.iloc[1:]) / dt

def min_av(db: pd.DataFrame) -> float:
    return angular_velocity(db).min()

def max_av(db: pd.DataFrame) -> float:
    return angular_velocity(db).max()

def mean_av(db: pd.DataFrame) -> float:
    return angular_velocity(db).mean()

def std_av(db: pd.DataFrame) -> float:
    return angular_velocity(db).std()

def max_min_av(db: pd.DataFrame) -> float:
    return max_av(db) - min_av(db)


def duration_of_movement(db: pd.DataFrame) -> float:
    return db.time.iloc[-1] - db.time.iloc[0]


def traveled_distance(db: pd.DataFrame) -> float:
    return np.hypot(db.x.diff().values, db.y.diff().values)[1:].sum()


def straightness(db: pd.DataFrame) -> float:
    return ((db.x.iloc[0] - db.x.iloc[-1]) ** 2 +
            (db.y.iloc[0] - db.y.iloc[-1]) ** 2) ** 0.5 / traveled_distance(db)


def TCM(db: pd.DataFrame) -> float:  # Trajectory Center of Mass
    return ((((db.x.iloc[:-1].values - db.x.iloc[1:].values) ** 2 +
              (db.y.iloc[:-1].values - db.y.iloc[1:].values) ** 2) ** 0.5) *
            db.time.iloc[1:]).sum() / traveled_distance(db)


def SC(db: pd.DataFrame) -> float:  # Scattering Coefficient
    return ((((db.x.iloc[:-1].values - db.x.iloc[1:].values) ** 2 +
              (db.y.iloc[:-1].values - db.y.iloc[1:].values) ** 2) ** 0.5) *
            db.time.iloc[1:] ** 2).sum() / traveled_distance(db) - TCM(db) ** 2


def M3(db: pd.DataFrame) -> float:  # Third Moment
    return ((((db.x.iloc[:-1].values - db.x.iloc[1:].values) ** 2 +
              (db.y.iloc[:-1].values - db.y.iloc[1:].values) ** 2) ** 0.5) *
            db.time.iloc[1:] ** 3).sum() / traveled_distance(db)


def M4(db: pd.DataFrame) -> float:  # Fourth Moment
    return ((((db.x.iloc[:-1].values - db.x.iloc[1:].values) ** 2 +
              (db.y.iloc[:-1].values - db.y.iloc[1:].values) ** 2) ** 0.5) *
            db.time.iloc[1:] ** 4).sum() / traveled_distance(db)


def TCrv(db: pd.DataFrame) -> float:  # Trajectory Curvature
    return ((velocity_x(db) * acceleration_y(db) -
             velocity_y(db) * acceleration_x(db)) / (velocity(db) ** 3)).mean()


def VCrv(db: pd.DataFrame) -> float:  # Velocity Curvature
    return (jerk(db) / (1 + acceleration(db) ** 2) ** (3/2)).mean()

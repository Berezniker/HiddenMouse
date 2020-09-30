from enum import Enum
import os

# path to the project
# ! CHANGE THIS !
ROOT_PATH = r"C:\Users\Alexey\Desktop\study\Diplom\HiddenMouse"
# ! CHANGE THIS !

# log file name
# src.utils.describe_data
LOG_JSON_FILE_NAME = "data_description.json"

# filename of the combined user sessions
# src.utils.combine_sessions
COMBINE_SESSION_NAME = "session_all.csv"

# time limit for feature extraction
# src.feature_extraction.feature_extractor.split_dataframe()
TIME_THRESHOLD = 3.0

# number of features
N_FEATURES = 17

# path to original datasets
# src.data_preprocessing
ORIGINAL_DATASET_PATH = os.path.join(ROOT_PATH, "original_dataset")

# path to preprocessed datasets, selected features and log files
DATASET_PATH = os.path.join(ROOT_PATH, "dataset")

# path to the log file
LOG_JSON_FILE_PATH = os.path.join(DATASET_PATH, LOG_JSON_FILE_NAME)

# list of names of all datasets
ALL_DATASET_NAME = ["BALABIT", "CHAOSHEN", "DATAIIT", "DFL", "TWOS"]

# list of paths to all datasets
ALL_DATASET_PATH = [os.path.join(DATASET_PATH, dataset_name)
                    for dataset_name in ALL_DATASET_NAME]

# neural network type
class NN(Enum):
    AUTOENCODER = 1
    CNN = 2

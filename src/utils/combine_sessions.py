import utils.constants as const
import pandas as pd
import numpy as np
import glob
import time
import os


def combine_sessions(datasets: list = None,
                     verbose: int = 0) -> None:
    """
    Combine user session into one file

    :param datasets: list of dataset names,
                     if None all datasets are used
    :param verbose: verbose output to stdout,
                    0 -- silence, [1, 2, 3] -- more verbose
    :return: None
    """
    if datasets is None:
        datasets = glob.glob(os.path.join(const.DATASET_PATH, '*'))
    else:
        datasets = [os.path.join(const.DATASET_PATH, data) for data in datasets]

    for dataset_path in datasets:
        if verbose >= 1: print(f"> {os.path.basename(dataset_path)}")
        for feature_path in glob.glob(os.path.join(dataset_path, '*features')):
            if verbose >= 2: print(f">> {os.path.basename(feature_path)}")
            if verbose >= 3: print(">>>", end=' ')
            for user_path in glob.glob(os.path.join(feature_path, 'user*')):
                if verbose >= 3: print(os.path.basename(user_path), end=' ')
                features = None
                for session_path in glob.glob(os.path.join(user_path, 'session_[0-9]*')):
                    f = pd.read_csv(session_path).values
                    features = np.vstack((features, f)) if features is not None else f
                save_path = os.path.join(user_path, const.COMBINE_SESSION_NAME)
                pd.DataFrame(features).to_csv(save_path, index=False, header=False)
            if verbose >= 3: print()


if __name__ == '__main__':
    start_time = time.time()
    print('Run!')
    combine_sessions(verbose=3)
    print(f'End, time: {time.time() - start_time:.3f} sec')

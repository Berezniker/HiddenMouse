import pandas as pd
import numpy as np
import glob
import time
import os


def combine_sessions(root_directory: str = '../../dataset',
                     final_name: str = 'session_all',
                     verbose: int = 0) -> None:
    for dataset_path in glob.glob(os.path.join(root_directory, '*')):
        if verbose >= 1: print(f"> {os.path.basename(dataset_path)}")
        for feature_path in glob.glob(os.path.join(dataset_path, '*features')):
            if verbose >= 2: print(f">> {os.path.basename(feature_path)}")
            if verbose >= 3: print(">>>", end=' ')
            for user_path in glob.glob(os.path.join(feature_path, 'user*')):
                if verbose >= 3: print(os.path.basename(user_path), end=' ')
                features = None
                for session_path in glob.glob(os.path.join(user_path, 'session_[0-9]*')):
                    f = pd.read_csv(session_path, sep=',').values
                    features = np.vstack((features, f)) if features is not None else f
                save_path = os.path.join(user_path, final_name)
                pd.DataFrame(features).to_csv(save_path, index=False, header=False)
            if verbose >= 3: print()


if __name__ == '__main__':
    start_time = time.time()
    print('Run!')
    combine_sessions(verbose=3)
    print(f'End, time: {time.time() - start_time:.3f} sec')

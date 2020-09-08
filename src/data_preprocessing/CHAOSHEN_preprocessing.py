from data_preprocessing.general_preprocessing import *
from utils.color import COLOR
from itertools import count
import pandas as pd
import time
import glob
import os

######################################################################
# CHAOSHEN dataset
#    |
#   ...
######################################################################


def chaoshen_preprocessing(data_dir: str = "../../original_dataset/CHAOSHEN_original",
                           save_dir: str = "../../dataset/CHAOSHEN",
                           verbose: int = 0) -> None:
    rename_fields = {}
    drop_fields = []
    clear_directory(save_dir)

    for user_path in glob.glob(os.path.join(data_dir, '*')):
        user_name = 'user' + os.path.basename(user_path)
        if verbose >= 2: print(f">> {user_name}")
        for session_path in glob.glob(os.path.join(user_path, '*.ef')):
            pass


if __name__ == '__main__':
    start_time = time.time()
    print(f"{COLOR['italics']}CHAOSHEN{COLOR['none']} Run!")
    chaoshen_preprocessing(verbose=3)
    print(f"run time: {time.time() - start_time:.3f}")

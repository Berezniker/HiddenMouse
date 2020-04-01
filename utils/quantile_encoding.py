from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import glob
import time
import os


def quantile_encoding(root_directory: str = '../../dataset',
                      dataset: str = None,
                      final_name: str = 'session_all_quantile',
                      n_quantiles: int = 4,
                      verbose: int = 0) -> None:
    final_name = f"{final_name}_{n_quantiles}"
    dataset = dataset if dataset is not None else '*'
    for dataset_path in glob.glob(os.path.join(root_directory, dataset)):
        if verbose >= 1: print(f"> {os.path.basename(dataset_path)}")
        for feature_path in glob.glob(os.path.join(dataset_path, 'train_features')):
            if verbose >= 3: print(">>>", end=' ')
            for user_path in glob.glob(os.path.join(feature_path, 'user*')):
                if verbose >= 3: print(os.path.basename(user_path), end=' ')
                train_path = user_path
                test_path = user_path.replace('train', 'test')
                train_features = pd.read_csv(os.path.join(train_path, 'session_all'), header=None)
                test_features = pd.read_csv(os.path.join(test_path, 'session_all'), header=None)

                for (name, f) in train_features.iteritems():
                    train_quant = pd.qcut(f, q=n_quantiles, duplicates='drop')
                    le = LabelEncoder().fit(train_quant)
                    train_features[name] = le.transform(train_quant)
                    # TODO can I do something better?
                    def foo(x):
                        return np.array([(x in c) for c in le.classes_]).argmax()
                    test_features[name] = test_features[name].apply(foo)

                train_features.to_csv(os.path.join(train_path, final_name), header=False, index=False)
                test_features.to_csv(os.path.join(test_path, final_name), header=False, index=False)
            if verbose >= 3: print()


if __name__ == '__main__':
    start_time = time.time()
    print('Run!')
    quantile_encoding(verbose=3, n_quantiles=10)
    print(f'End, time: {time.time() - start_time:.3f} sec')

from math import ceil
import pandas as pd
import numpy as np
import glob
import os


def database_preprocessing(db, mode):
    size_before = db.index.size
    if mode == 'BALABIT':
        db.rename({'client timestamp': 'time'}, axis=1, inplace=True)
        db.drop(db[db.button == 'Scroll'].index, axis=0, inplace=True)
    db.drop_duplicates(inplace=True)
    db.drop_duplicates(subset='time', inplace=True)
    diff = (db[['x', 'y']] - db[['x', 'y']].shift(1))
    db.drop(db[np.all(diff == 0, axis=1)].index, inplace=True)
    db.drop(db[db.x > 2000].index, inplace=True)
    db.drop(db[db.y > 1200].index, inplace=True)

    return 1 - (db.index.size / size_before)


def count_number_of_records_for_each_user(data_path):
    n_records = list()
    for user in glob.glob(os.path.join(data_path, 'user*')):
        n_features = 0
        for session in glob.glob(os.path.join(user, 'session*')):
            n_features += pd.read_csv(session).index.size
        n_records.append(n_features)
    return np.asarray(n_records)


def count_time_for_each_user(data_path):
    all_time = list()
    time_field = 'time' if 'dataset2' in data_path else 'client timestamp'
    for user in glob.glob(os.path.join(data_path, 'user*')):
        time = 0
        for session in glob.glob(os.path.join(user, 'session*')):
            time += pd.read_csv(session)[time_field].iloc[-1]
        all_time.append(time)
    return np.asarray(all_time)


def count_resizing_after_preprocessing_for_each_user(data_path):
    emission_percentage = list()
    mode = 'DATAIIT' if 'dataset2' in data_path else 'BALABIT'
    for user in glob.glob(os.path.join(data_path, 'user*')):
        temp = list()
        for session in glob.glob(os.path.join(user, 'session*')):
            temp.append(database_preprocessing(pd.read_csv(session), mode))
        emission_percentage.append(np.mean(temp))
    return np.asarray(emission_percentage)


def run_mean_std_of_records_for_dataset(dataset):
    train_n_records = count_number_of_records_for_each_user(f'../{dataset}/train_files')
    test_n_records = count_number_of_records_for_each_user(f'../{dataset}/test_files')
    mean_n_records = ceil(np.mean(train_n_records + test_n_records))
    std_n_records = ceil(np.std(train_n_records + test_n_records))
    print(f"{dataset}.n_records:\n"
          f"  mean: {mean_n_records}, std: {std_n_records}")


def run_mean_std_of_records_for_feature(feature, sec, act='5'):
    train_n_records = count_number_of_records_for_each_user(f'../{feature}/n_features_81/{sec}sec_{act}act/train_features')
    mean_n_records = ceil(np.mean(train_n_records))
    std_n_records = ceil(np.std(train_n_records))
    print(f"{feature}.n_records.{sec}sec:\n"
          f"  mean: {mean_n_records}, std: {std_n_records}")


def run_mean_std_of_time_for_dataset(dataset):
    train_times = count_time_for_each_user(f'../{dataset}/train_files')
    test_times = count_time_for_each_user(f'../{dataset}/test_files')
    mean_time = ceil(np.mean(train_times + test_times) / 3600)
    std_time = ceil(np.std(train_times + test_times) / 3600)
    print(f"{dataset}.time:\n"
          f"  mean: {mean_time}, std: {std_time}")


def run_mean_std_percent_outliers_for_dataset(dataset):
    train_outliers = count_resizing_after_preprocessing_for_each_user(f'../{dataset}/train_files')
    test_outliers = count_resizing_after_preprocessing_for_each_user(f'../{dataset}/test_files')
    mean_outliers = np.mean((train_outliers + test_outliers)/2)
    std_outliers = np.std((train_outliers + test_outliers)/2)
    print(f"{dataset}.outliers:\n"
          f"  mean: {mean_outliers:.2f}, std: {std_outliers:.2f}")


if __name__ == '__main__':
    # run_mean_std_of_records_for_dataset('dataset')
    # run_mean_std_of_time_for_dataset('dataset')
    # run_mean_std_percent_outliers_for_dataset('dataset')
    run_mean_std_of_records_for_feature('features', sec='2', act='3')
    run_mean_std_of_records_for_feature('features', sec='5')
    run_mean_std_of_records_for_feature('features', sec='10')
    run_mean_std_of_records_for_feature('features', sec='15')

    # run_mean_std_of_records_for_dataset('dataset2')
    # run_mean_std_of_time_for_dataset('dataset2')
    # run_mean_std_percent_outliers_for_dataset('dataset2')
    run_mean_std_of_records_for_feature('features2', sec='2')
    run_mean_std_of_records_for_feature('features2', sec='5')
    run_mean_std_of_records_for_feature('features2', sec='10')
    run_mean_std_of_records_for_feature('features2', sec='15')

import utils.constants as const
import json
import glob
import time
import os


def create_new_log(save_old_log_file: bool = True) -> dict:
    """
    Create new json log file

    :param save_old_log_file: if <True> saves old json log file
    :return: empty dictionary
    """
    if save_old_log_file and os.path.exists(const.LOG_JSON_FILE_PATH):
        hash_value = hash(time.ctime(time.time())[4:]) % 1000000
        old_log_path = const.LOG_JSON_FILE_PATH.replace(".json", f"_{hash_value}.json")
        os.rename(src=const.LOG_JSON_FILE_PATH, dst=old_log_path)
    return dict()


def load_log() -> dict:
    """
    Upload json log file

    :return: dictionary with data description
    """
    describer = dict()
    if os.path.exists(const.LOG_JSON_FILE_PATH):
        with open(const.LOG_JSON_FILE_PATH, mode="r") as f:
            describer = json.load(fp=f)
    return describer


def dump_log(describer: dict) -> None:
    """
    Save json log file

    :param describer: dictionary with data description
    :return: None
    """
    with open(const.LOG_JSON_FILE_PATH, mode="w") as f:
        json.dump(obj=describer, fp=f, ensure_ascii=False, indent=4)


def count_files_and_lines(user_path: str) -> (int, int):
    """
    Count the number of files and lines in files for the user

    :param user_path: path to user directory
    :return: (number_of_files, number_of_lines)
    """
    n_files, n_lines = 0, 0

    for path in glob.iglob(os.path.join(user_path, 'session_[0-9]*.csv')):
        n_files += 1
        n_lines += sum(1
                       for line in open(path, mode='r')
                       if line and not line.isspace())

    return n_files, n_lines


def describe_data() -> None:
    """
    Form & save descriptive statistics on the data in json format:

    {
        DATASET_NAME: {
            USER_NAME: {
                ...
                'total_number_of_features': int,
                'total_number_of_files': int,
                'total_number_of_lines': int,
                'ratio_train_test': float
            },
            ...
            'number_of_users': int
        },
        ...
    }

    :return: dictionary with data description
    """
    describer = load_log()
    describer["_time_threshold"] = const.TIME_THRESHOLD

    total_n_users = 0
    for dataset_path in glob.iglob(os.path.join(const.DATASET_PATH, '*')):
        if not os.path.isdir(dataset_path):
            continue
        dataset_name = os.path.basename(dataset_path)
        describer.setdefault(dataset_name, dict())
        print(dataset_name)

        n_users = 0
        for user_path in glob.iglob(os.path.join(dataset_path, 'train_files/user*')):
            n_users += 1
            user_name = os.path.basename(user_path)
            describer[dataset_name].setdefault(user_name, dict())

            path = os.path.join('train_files', user_name)
            path = os.path.join(dataset_path, path)
            n_train_files, n_train_lines = count_files_and_lines(path)

            path = os.path.join('test_files', user_name)
            path = os.path.join(dataset_path, path)
            n_test_files, n_test_lines = count_files_and_lines(path)

            path = os.path.join('t*_features', user_name)
            path = os.path.join(dataset_path, path)
            _, n_lines = count_files_and_lines(path)
            describer[dataset_name][user_name]['total_number_of_features'] = n_lines

            describer[dataset_name][user_name]['total_number_of_files'] = n_train_files + n_test_files
            describer[dataset_name][user_name]['total_number_of_lines'] = n_train_lines + n_test_lines
            ratio = round(n_train_lines / (n_train_lines + n_test_lines), 2)
            describer[dataset_name][user_name]['ratio_train_test'] = ratio

        describer[dataset_name]['number_of_users'] = n_users
        total_n_users += n_users
    describer['total_number_of_users'] = total_n_users
    dump_log(describer)


if __name__ == '__main__':
    print("Run!")
    start_time = time.time()
    describe_data()
    print(f"End of run. Time: {(time.time() - start_time) / 60:.1f} min")
    # 7 min

from utils import describe_data
import utils.constants as const
import subprocess
import datetime
import time
import os


def all_preprocessing() -> None:
    """
    Preprocess all data

    :return: None
    """
    describer = describe_data.create_new_log()

    for dataset in const.ALL_DATASET_NAME:
        dataset_time = time.time()
        describer.setdefault(dataset, dict())
        print(f"> {dataset:<8}", end=' ')

        proc = subprocess.run(
            args=f"python {dataset}_preprocessing.py",
            input="Y",  # "YES" for general_preprocessing.clear_directory()
            stdout=subprocess.PIPE,
            encoding='ascii',
            shell=True)

        save_path = os.path.join(const.DATASET_PATH, dataset)
        save_path = os.path.join(save_path, f"log_{dataset}_preprocessing.txt")
        with open(save_path, mode='w') as f:
            f.write(f"{datetime.datetime.now().isoformat(sep=' ')[:-7]}\n\n")
            f.write(proc.stdout)

        dataset_time = round((time.time() - dataset_time) / 60.0, 1)
        describer[dataset]["preprocessing_time_(min)"] = dataset_time
        print(f"{dataset_time:4.1f} min")

    describe_data.dump_log(describer)


if __name__ == '__main__':
    print('Run!')
    start_time = time.time()
    all_preprocessing()
    print(f'End of run. time: {(time.time() - start_time) / 60.0:.1f} min')

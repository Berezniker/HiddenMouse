from utils import describe_data
import subprocess
import datetime
import time
import glob
import os


def all_preprocessing() -> None:
    """
    Preprocess all data

    :return: None
    """
    all_dataset = [os.path.basename(path)
                   for path in glob.iglob("../../dataset/*") if os.path.isdir(path)]
    describer = describe_data.create_new_log()

    for dataset in all_dataset:
        dataset_time = time.time()
        describer.setdefault(dataset, dict())
        print(f"> {dataset:<8}", end=' ')

        proc = subprocess.run(
            args=f"python {dataset}_preprocessing.py",
            input="Y",  # "YES" for general_preprocessing.clear_directory()
            stdout=subprocess.PIPE,
            encoding='ascii',
            shell=True)

        with open(f"../../dataset/{dataset}/log_{dataset}_preprocessing.txt", mode='w') as f:
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

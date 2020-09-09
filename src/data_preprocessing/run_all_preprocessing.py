import subprocess
import datetime
import time


def all_preprocessing(datasets: list) -> None:
    for dataset in datasets:
        dataset_time = time.time()
        print(f"> {dataset:7}", end=' ')
        proc = subprocess.run(
            args=f"python {dataset}_preprocessing.py",
            input="Y",
            stdout=subprocess.PIPE,
            encoding='ascii',
            shell=True)
        with open(f"../../dataset/{dataset}/log_{dataset}_preprocessing.txt", mode='w') as f:
            f.write(f"{datetime.datetime.now().isoformat(sep=' ')[:-7]}\n\n")
            f.write(proc.stdout)
        print(f"{(time.time() - dataset_time) / 60.0:4.1f} min")


if __name__ == '__main__':
    print('Run!')
    start_time = time.time()
    all_datasets = ["BALABIT", "CHAOSHEN", "DATAIIT", "TWOS", "DFL"]
    all_preprocessing(all_datasets)
    print(f'End of run. time: {(time.time() - start_time) / 60.0:.1f} min')

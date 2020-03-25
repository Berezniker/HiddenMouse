import subprocess
import time

def all_preprocessing() -> None:
    proc = subprocess.Popen("python BALABIT_preprocessing.py", shell=True)
    proc.wait()
    proc = subprocess.Popen("python DATAIIT_preprocessing.py", shell=True)
    proc.wait()
    proc = subprocess.Popen("python TWOS_preprocessing.py", shell=True)
    proc.wait()


if __name__ == '__main__':
    print('Run!')
    start_time = time.time()
    all_preprocessing()
    print(f'End of run. time: {time.time() - start_time:.3f}')

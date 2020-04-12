import feature_extraction as extractor
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from itertools import count
import pandas as pd
import numpy as np
import time as t
import glob
import os

TIME_THRESHOLD = 5
LABELS = None


def pyplot_image(db, mode='show', background='dark_background', gradline=True,
                 cmap='plasma', save_path='./', image_name='image', fmt='png'):
    plt.style.use(background)
    fig, ax = plt.subplots(figsize=(20, 12))

    x, y = db.x.values, db.y.values
    dydx = np.hypot((x[:-1] - x[1:]), (y[:-1] - y[1:]))
    dydx = np.insert(dydx, 0, dydx[0]).clip(0, 100)

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    if dydx.max() < 100:
        # simulate the outlier point
        x, y = np.append(x, 5000), np.append(y, 5000)
        dydx = np.append(dydx, 100)

    ax.scatter(x=x, y=y, s=16, c=dydx, marker='o', cmap=cmap, zorder=2)

    if not gradline:
        ax.plot(db.x.values, db.y.values, color='whitesmoke', linewidth=9, zorder=1)
    else:
        lc = LineCollection(segments, cmap=cmap)
        lc.set_array(dydx)
        lc.set_linewidth(4)
        lc.set_zorder(1)
        line = ax.add_collection(lc)
        # fig.colorbar(line, ax=ax)

    # configuration
    ax.set_xlim([-50, 2050])
    ax.set_ylim([-50, 1250])
    ax.grid(False)
    ax.tick_params(labelleft=False, labelbottom=False)
    plt.gca().axison = False

    if mode == 'show':
        plt.show()
    elif mode == 'save':
        plt.savefig(f"{save_path}/{image_name}.{fmt}", dpi=20, format=fmt,
                    transparent=False, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def create_images(database, save_images=False, session_name='session', save_path='./temp', fmt='png',
                  only_one_image=False):
    extractor.database_preprocessing(database, isbalabit=('BALABIT' in save_path))

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    sequence = count()
    mode = 'save' if save_images else 'show'
    for segment in extractor.split_dataframe(database, time_threshold=TIME_THRESHOLD):
        unique_num = next(sequence)
        pyplot_image(db=segment, mode=mode, save_path=save_path,
                     image_name=f'{session_name}_{unique_num:04}', fmt=fmt)
        if unique_num and unique_num % 500 == 0:
            print(f'...{unique_num}', end='')
        if only_one_image:
            print('>>>> [!] only_one_feature')
            break
    else:
        print(f'...{unique_num}.')


def run(data_path, save_path, save_images=False,
        only_one_user=False, only_one_session=False, only_one_image=False):
    global LABELS

    for user in glob.glob(os.path.join(data_path, 'user*')):
        user_start_time = t.time()
        user_name = os.path.basename(user)
        print(f'> {user_name}')
        n_session = len(list(os.listdir(user)))
        for i, session in enumerate(glob.glob(os.path.join(user, 'session*')), 1):
            session_name = os.path.basename(session)
            if LABELS is not None and session_name not in LABELS.filename.values:
                continue  # skip illegal test records for BALABIT
            session_start_time = t.time()
            save_img_path = f'{save_path}/{user_name}/movement'
            # vvvvvv
            create_images(database=pd.read_csv(session), save_images=save_images, session_name=session_name,
                          save_path=save_img_path, only_one_image=only_one_image)
            # ^^^^^^
            print(f'>> [{user_name[-2:]}.{i:03}/{n_session}] '
                  f'{session_name} time: {t.time() - session_start_time:6.3f} sec')
            if only_one_session:
                print('>> [!] only_one_session')
                break
        print(f'> {user_name} time: {t.time() - user_start_time:.3f} sec\n')
        if only_one_user or only_one_session:
            print('> [!] only_one_user')
            break


if __name__ == '__main__':
    start_time = t.time()
    print(f"SPLIT_TIME_THRESHOLD = {TIME_THRESHOLD}")

    print('train: RUN!')
    train_start_time = t.time()
    run(data_path='../dataset/train_files', save_images=True,
        save_path=f'../images/BALABIT/{TIME_THRESHOLD}sec/train_features')
    print(f'train_time: {t.time() - train_start_time:.3f} sec\n')  # 2.2 hours

    print('test: RUN!')
    test_start_time = t.time()
    LABELS = pd.read_csv(r'../dataset/labels.csv')
    run(data_path='../dataset/test_files', save_images=True,
        save_path=f'../images/BALABIT/{TIME_THRESHOLD}sec/test_features')
    LABELS = None
    print(f'test_time: {t.time() - test_start_time:.3f} sec\n')  # 1.4 hours

    print('DATAIIT: RUN!')
    dataiit_start_time = t.time()
    run(data_path='../dataset2/train_files', save_images=True,
        save_path=f'../images/DATAIIT/{TIME_THRESHOLD}sec/train_features')
    print(f'DATAIIT_time: {t.time() - dataiit_start_time:.3f} sec\n')  # 3.3 hours

    print(f'main_time: {t.time() - start_time:.3f} sec')  # 7 hours

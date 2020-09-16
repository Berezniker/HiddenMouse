from pynput import mouse
import datetime
import time
import csv
import os

def run(username: str = 'Unknown') -> None:
    header = ["timestamp", "button", "state", "x", "y"]
    dir_name = f"../../original_dataset/MY_original/User_{username}"
    os.makedirs(dir_name, exist_ok=True)
    date = datetime.datetime.now().isoformat(sep='_')[:-7].replace(':', '-')
    log_name = f"mouse_logger_{username}_{date}.csv"

    with open(os.path.join(dir_name, log_name), mode='w', newline='') as csvfile:
        logger = csv.writer(csvfile, delimiter=',')
        logger.writerow(header)
        start_time = time.time()
        n_row, max_row = 0, 100_000

        def improve_xy(x: int, y: int):
            x = x if x > 0 else 0
            y = y if y > 0 else 0
            return x, y

        def check_end():
            nonlocal n_row
            n_row += 1
            return n_row < max_row

        def on_move(x, y):
            x, y = improve_xy(x, y)
            logger.writerow([
                f"{time.time() - start_time}",
                f"NoButton",
                f"Move",
                f"{x}",
                f"{y}"
            ])
            return check_end()

        def on_click(x, y, button, pressed):
            x, y = improve_xy(x, y)
            state = "Pressed" if pressed else "Released"
            logger.writerow([
                f"{time.time() - start_time}",
                f"{button.name.capitalize()}",
                f"{state}",
                f"{x}",
                f"{y}"
            ])
            return check_end()

        def on_scroll(x, y, dx, dy):
            x, y = improve_xy(x, y)
            state = "Down" if dy < 0 else "Up"
            logger.writerow([
                f"{time.time() - start_time}",
                f"Scroll",
                f"{state}",
                f"{x}",
                f"{y}"
            ])
            return check_end()

        with mouse.Listener(
                on_move=on_move,
                on_click=on_click,
                on_scroll=on_scroll) as listener:
            listener.join()
            # stopped when the chek_end() returns False


if __name__ == "__main__":
    run("Alexey")

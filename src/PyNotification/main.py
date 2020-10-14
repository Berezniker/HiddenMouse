import win10toast
import platform
import enum
import time
import os


# Windows block:
# import ctypes
# ctypes.windll.user32.LockWorkStation()

# compile:
# pyinstaller --onefile --clean --noconsole --icon=alarm.ico main.py


class OSType(enum.Enum):
    Windows = 0
    Linux = 1
    Mac = 2


def get_os_type() -> OSType:
    """
    returns the operating system type: OneOf[Windows, Linux, Mac]

    :return: OSType
    """
    os_name = platform.system()
    if os_name == "Windows":
        return OSType.Windows
    elif os_name == "Linux":
        return OSType.Linux
    elif os_name == "Darwin":
        return OSType.Mac
    else:
        raise RuntimeError(f"Unknown OS name: platform.system() -> '{os_name}'")


class Notification:
    def __init__(self):
        self._os_type = get_os_type()

    def send(self, title: str,
             message: str,
             duration: float = 5.0) -> None:
        """
        Displays notifications in the taskbar

        :param title: the headline of the message
        :param message: message text
        :param duration: duration of notification
        :return: None
        """
        if self._os_type == OSType.Windows:
            win10toast.ToastNotifier().show_toast(
                title=title,
                msg=message,
                icon_path="alarm.ico",
                duration=duration
            )

        elif self._os_type == OSType.Mac:
            command = f'''osascript -e 'display notification "{message}" with title "{title}"'''
            os.system(command)
            time.sleep(duration)
        elif self._os_type == OSType.Linux:
            command = f'notify-send "{title}" "{message}"'
            os.system(command)
            time.sleep(duration)


if __name__ == '__main__':
    print("Run!")
    Notification().send(
        title="Attention!",
        message="The system will be locked after 5 seconds ...",
        duration=5
    )
    print("End.")

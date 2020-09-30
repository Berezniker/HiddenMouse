import utils.constants as const
import datetime
import json
import glob
import os


def describe_project(project_path: str) -> dict:
    """
    Count lines of project code

    :param project_path: path to the project
    :return: dictionary
    """
    describer = dict()
    project_n_lines = 0

    for dir_path in glob.iglob(os.path.join(project_path, '*')):
        dir_name = os.path.basename(dir_path)
        describer[dir_name] = dict()
        dir_n_lines = 0

        for file_path in glob.iglob(os.path.join(dir_path, '*')):
            if file_path.endswith('.py'):
                n_lines = sum(1 for _ in open(file_path, mode='rb'))
                describer[dir_name][os.path.basename(file_path)] = n_lines
                dir_n_lines += n_lines

        describer[dir_name]["_total_lines"] = dir_n_lines
        project_n_lines += dir_n_lines

    describer["__total_lines"] = project_n_lines
    describer["latest_update_date"] = datetime.datetime.now().isoformat(sep=' ')[:-7]
    return {os.path.basename(project_path): describer}


if __name__ == '__main__':
    path = os.path.join(const.ROOT_PATH, "src")
    describer = describe_project(path)
    with open("./project_description.json", mode='w') as f:
        json.dump(obj=describer, fp=f, ensure_ascii=True, indent=4)

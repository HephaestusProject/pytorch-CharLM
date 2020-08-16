from pathlib import Path


def get_next_version(root_dir: Path):
    version_prefix = "v"
    if not root_dir.exists():
        next_version = 0
    else:
        existing_versions = []
        for child_path in root_dir.iterdir():
            if child_path.is_dir() and child_path.name.startswith(version_prefix):
                existing_versions.append(int(child_path.name[len(version_prefix) :]))

        if len(existing_versions) == 0:
            last_version = -1
        else:
            last_version = max(existing_versions)

        next_version = last_version + 1
    return f"{version_prefix}{next_version:0>3}"


class IntList(list):
    def __init__(self, arg):
        int_list = [int(value) for value in arg.split(",")]
        super(IntList, self).__init__(int_list)


class StickingProgressBarCallback(Callback):
    def __init__(self):
        super().__init__()

    def on_epoch_start(self, trainer, pl_module):
        print(" ")

from pathlib import Path

def next_version_dir_name(root_dir: Path):
    version_prefix = "version_"
    if not root_dir.exists():
        next_version = 0
    else:
        existing_versions = []
        for child_path in root_dir.iterdir():
            if child_path.is_dir() and child_path.name.startswith(version_prefix):
                existing_versions.append(int(child_path.name.split("_")[1]))

        if len(existing_versions) == 0:
            last_version = -1
        else:
            last_version = max(existing_versions)

        next_version = last_version + 1
    return f"{version_prefix}{next_version:0>3}"

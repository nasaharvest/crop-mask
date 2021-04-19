from pathlib import Path
import subprocess


def get_dvc_dir(dvc_dir_name: str) -> Path:
    dvc_dir = Path(__file__).parent.parent / f"data/{dvc_dir_name}"
    if not dvc_dir.exists():
        subprocess.run(["dvc", "pull", f"data/{dvc_dir_name}"], check=True)
        if not dvc_dir.exists():
            raise FileExistsError(f"{str(dvc_dir)} was not found.")
        if not any(dvc_dir.iterdir()):
            raise FileExistsError(f"{str(dvc_dir)} should not be empty.")
    return dvc_dir

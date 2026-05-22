import shutil
import subprocess
from pathlib import Path


def convert_to_training_wav(raw_path: Path, wav_path: Path) -> bool:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return False

    wav_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-y",
        "-i",
        str(raw_path),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-sample_fmt",
        "s16",
        str(wav_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    return result.returncode == 0 and wav_path.exists() and wav_path.stat().st_size > 0

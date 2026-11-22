"""Download the pinned source archive using the build Python's network configuration."""
from pathlib import Path
import shutil
import sys
from urllib.request import urlopen


def main():
    url, destination = sys.argv[1:]
    archive = Path(destination)
    if archive.exists():
        return
    temporary = archive.with_suffix(".part")
    with urlopen(url, timeout=120) as response, temporary.open("wb") as stream:
        shutil.copyfileobj(response, stream)
    temporary.replace(archive)


if __name__ == "__main__":
    main()

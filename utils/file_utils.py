"""
====================================================
RetinaSense

File Utility Functions


Description:
Common file and folder operations used
throughout the RetinaSense project.

====================================================
"""

from pathlib import Path
import shutil

from core.logger import logger


# ====================================================
# Create Directory
# ====================================================

def create_directory(directory: Path):

    """
    Create a directory if it doesn't exist.
    """

    directory.mkdir(parents=True, exist_ok=True)

    logger.info(f"Directory Ready : {directory}")


# ====================================================
# Check Directory
# ====================================================

def directory_exists(directory: Path):

    return directory.exists()


# ====================================================
# Copy File
# ====================================================

def copy_file(source: Path, destination: Path):

    shutil.copy2(source, destination)

    logger.info(f"Copied : {source.name}")


# ====================================================
# Move File
# ====================================================

def move_file(source: Path, destination: Path):

    shutil.move(source, destination)

    logger.info(f"Moved : {source.name}")


# ====================================================
# Delete File
# ====================================================

def delete_file(file_path: Path):

    if file_path.exists():

        file_path.unlink()

        logger.info(f"Deleted : {file_path.name}")


# ====================================================
# Count Files
# ====================================================

def count_files(directory: Path):

    return len([file for file in directory.iterdir() if file.is_file()])


# ====================================================
# Count Images
# ====================================================

def count_images(directory: Path):

    image_extensions = [
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tif",
        ".tiff"
    ]

    total = 0

    for ext in image_extensions:

        total += len(list(directory.rglob(f"*{ext}")))

    return total


# ====================================================
# List Images
# ====================================================

def list_images(directory: Path):

    image_extensions = [
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tif",
        ".tiff"
    ]

    images = []

    for ext in image_extensions:

        images.extend(directory.rglob(f"*{ext}"))

    return sorted(images)
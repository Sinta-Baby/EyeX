from pathlib import Path

from utils.file_utils import *

from core.paths import *


print("=" * 60)
print("Testing File Utilities")
print("=" * 60)

# Test directory creation

test_folder = PROJECT_ROOT / "temp_test"

create_directory(test_folder)

print("\nDirectory Exists")

print(directory_exists(test_folder))

print("\nTotal Files")

print(count_files(test_folder))

print("\nDeleting Test Folder...")

test_folder.rmdir()

print("Done")
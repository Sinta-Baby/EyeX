from core.paths import *

print("=" * 60)
print("ODIR PATH TEST")
print("=" * 60)

paths = {
    "ODIR_DATASET": ODIR_DATASET,
    "ODIR_METADATA": ODIR_METADATA,
    "ODIR_IMAGES": ODIR_IMAGES,
}

for name, path in paths.items():

    print(f"\n{name}")
    print(path)
    print("Exists:", path.exists())
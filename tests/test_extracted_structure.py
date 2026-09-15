from core.paths import EXTRACTED_DATASET

print("=" * 60)
print("Extracted Dataset Structure")
print("=" * 60)

print("Extracted Path:")
print(EXTRACTED_DATASET)

print("\nContents:")

if EXTRACTED_DATASET.exists():
    for item in EXTRACTED_DATASET.iterdir():
        print(item)
else:
    print("EXTRACTED_DATASET folder does not exist!")
import random
import shutil
from pathlib import Path

IMAGE_DIR = Path('images')
LABEL_DIR = Path('labels')

TRAIN_RATIO = 0.8
RANDOM_SEED = 40

IMAGE_EXTENSIONS = {
    '.jpg',
    '.jpeg',
}

OUT_IMAGES_TRAIN = IMAGE_DIR / 'train'
OUT_IMAGES_VAL = IMAGE_DIR / 'val'

OUT_LABELS_TRAIN = LABEL_DIR / 'train'
OUT_LABELS_VAL = LABEL_DIR / 'val'

for directory in [
    OUT_IMAGES_TRAIN,
    OUT_IMAGES_VAL,
    OUT_LABELS_TRAIN,
    OUT_LABELS_VAL,
]:
    directory.mkdir(parents=True, exist_ok=True)

# Find image/label pairs

pairs = []

for image_path in IMAGE_DIR.iterdir():
    if (
        not image_path.is_file()
        or image_path.parent != IMAGE_DIR
        or image_path.suffix.lower() not in IMAGE_EXTENSIONS
    ):
        continue

    label_path = LABEL_DIR / f'{image_path.stem}.txt'

    if not label_path.exists():
        print(f'Warning: no label found for {image_path.name}')
        continue

    pairs.append((image_path, label_path))

if len(pairs) == 0:
    raise RuntimeError('No matching image/label pairs found.')

# Randomise and split

random.seed(RANDOM_SEED)
random.shuffle(pairs)

split_index = int(len(pairs) * TRAIN_RATIO)

train_pairs = pairs[:split_index]
val_pairs = pairs[split_index:]

# Copy files


def copy_pairs(dataset, image_dst, label_dst):
    for image_path, label_path in dataset:
        shutil.copy2(image_path, image_dst / image_path.name)
        shutil.copy2(label_path, label_dst / label_path.name)


copy_pairs(train_pairs, OUT_IMAGES_TRAIN, OUT_LABELS_TRAIN)
copy_pairs(val_pairs, OUT_IMAGES_VAL, OUT_LABELS_VAL)


print()
print('Dataset split complete.')
print(f'Total samples : {len(pairs)}')
print(f'Train samples : {len(train_pairs)}')
print(f'Val samples   : {len(val_pairs)}')

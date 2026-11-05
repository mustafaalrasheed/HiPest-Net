"""IP102 classification data pipeline.

Expected layout (any one of these is detected automatically):

    <root>/train.txt, val.txt, test.txt   and   <root>/images/{train,val,test}/
    <root>/classification/{train,val,test}.txt and <root>/classification/{train,val,test}/
    <root>/{train,val,test}.txt           and   <root>/classification/{train,val,test}/
    <root>/{train,val,test}.txt           and   <root>/images/   (official ip102_v1.1: one flat image folder)

Each annotation line is "<image file name> <label>". Labels may be 0-based or
1-based; 1-based labels are shifted automatically.
"""

import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

from config import IMAGENET_MEAN, IMAGENET_STD, NUM_CLASSES


def _resolve(root, split):
    ann_candidates = [
        os.path.join(root, f"{split}.txt"),
        os.path.join(root, "classification", f"{split}.txt"),
    ]
    img_candidates = [
        os.path.join(root, "images", split),
        os.path.join(root, "images"),
        os.path.join(root, "classification", split),
        os.path.join(root, split),
    ]
    ann = next((p for p in ann_candidates if os.path.isfile(p)), None)
    img = next((p for p in img_candidates if os.path.isdir(p)), None)
    if ann is None or img is None:
        raise FileNotFoundError(
            f"Could not locate IP102 '{split}' annotations/images under {root}. "
            f"Tried annotations {ann_candidates} and image folders {img_candidates}."
        )
    return ann, img


class IP102(Dataset):
    def __init__(self, root, split, transform=None, draft_size=512):
        ann, self.img_dir = _resolve(root, split)
        names, labels = [], []
        with open(ann, "r", encoding="utf-8") as fh:
            for line in fh:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                names.append(parts[0])
                labels.append(int(parts[-1]))
        labels = np.asarray(labels, dtype=np.int64)
        if labels.max() == NUM_CLASSES:  # 1-based labels
            labels = labels - 1
        assert labels.min() >= 0 and labels.max() < NUM_CLASSES, "label range does not match 102 classes"
        self.names, self.labels = names, labels
        self.transform = transform
        self.draft_size = draft_size

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        img = Image.open(os.path.join(self.img_dir, self.names[i]))
        if self.draft_size:
            img.draft("RGB", (self.draft_size, self.draft_size))  # fast JPEG down-scaled decode
        img = img.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, int(self.labels[i])

    def class_counts(self):
        return np.bincount(self.labels, minlength=NUM_CLASSES)


class CPB(Dataset):
    """Citrus Pest Benchmark, official split (Bollis et al.): <root> holds the images and
    pests_{train,validation,test}_original.csv with columns images, Mite, Negative."""

    def __init__(self, root, split, transform=None, draft_size=512):
        import csv
        csv_name = {"train": "train", "val": "validation", "test": "test"}[split]
        self.img_dir = root
        names, labels = [], []
        with open(os.path.join(root, f"pests_{csv_name}_original.csv"), newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                names.append(row["images"])
                labels.append(int(row["Mite"]))
        self.names, self.labels = names, np.asarray(labels, dtype=np.int64)
        self.transform, self.draft_size = transform, draft_size

    __len__ = IP102.__len__
    __getitem__ = IP102.__getitem__
    class_counts = IP102.class_counts


def make_dataset(root, split, transform):
    if os.environ.get("HIPEST_DATASET", "ip102").lower() == "cpb":
        return CPB(root, split, transform)
    return IP102(root, split, transform)


def build_transforms(img_size=224):
    train = T.Compose(
        [
            T.RandomResizedCrop(img_size, scale=(0.35, 1.0), interpolation=InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(),
            T.RandAugment(num_ops=2, magnitude=9),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            T.RandomErasing(p=0.25),
        ]
    )
    test = T.Compose(
        [
            T.Resize(int(img_size * 256 / 224), interpolation=InterpolationMode.BICUBIC),
            T.CenterCrop(img_size),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train, test


def class_balanced_sampler(labels, q: float = 0.0, seed: int = 0):
    """Sampler drawing class j with probability proportional to n_j**q.

    q = 1 reproduces instance-balanced sampling, q = 0 gives fully
    class-balanced sampling, q = 0.5 is square-root sampling.
    """
    counts = np.bincount(labels, minlength=NUM_CLASSES).astype(np.float64)
    per_sample = counts[labels] ** (q - 1.0)
    g = torch.Generator()
    g.manual_seed(seed)
    return WeightedRandomSampler(torch.as_tensor(per_sample), num_samples=len(labels), replacement=True, generator=g)


def seed_worker(worker_id):
    s = torch.initial_seed() % 2**32
    np.random.seed(s)
    random.seed(s)


def make_loader(ds, batch_size, train, workers=8, sampler=None, seed=0):
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=(train and sampler is None),
        sampler=sampler,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=train,
        persistent_workers=workers > 0,
        worker_init_fn=seed_worker,
        generator=g,
    )

"""Shared constants for HiPest-Net (IP102 taxonomy).

IP102 groups its 102 pest classes by the crop they damage. The label space is
assumed to be contiguous per crop, in the order below:

    rice 14 | corn 13 | wheat 9 | beet 8 | alfalfa 13 | vitis 16 | citrus 19 | mango 10

Check this against the class list shipped with your copy of the dataset before
training. If your copy orders the classes differently, edit GROUP_SIZES (and, if
the classes are not contiguous per crop, remap the labels in data.py).
"""

import os

GROUP_NAMES = ["rice", "corn", "wheat", "beet", "alfalfa", "vitis", "citrus", "mango"]
GROUP_SIZES = [14, 13, 9, 8, 13, 16, 19, 10]
if os.environ.get("HIPEST_DATASET", "ip102").lower() == "cpb":
    # Citrus Pest Benchmark (official split): binary mite-present / mite-absent, no crop taxonomy.
    GROUP_NAMES, GROUP_SIZES = ["all"], [2]
NUM_CLASSES = sum(GROUP_SIZES)  # 102 (IP102) or 2 (CPB)
NUM_GROUPS = len(GROUP_SIZES)  # 8 (IP102) or 1 (CPB)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def class_to_group(sizes=GROUP_SIZES):
    """Return a list mapping every class index to its crop-group index."""
    mapping = []
    for g, n in enumerate(sizes):
        mapping.extend([g] * n)
    return mapping


def group_slices(sizes=GROUP_SIZES):
    """Return (start, end) class-index ranges for every crop group."""
    out, start = [], 0
    for n in sizes:
        out.append((start, start + n))
        start += n
    return out

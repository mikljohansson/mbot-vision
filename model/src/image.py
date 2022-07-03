import random

import numpy as np


def get_input_box(mask, target_width, target_height):
    # Mask is HxW format
    height, width = mask.shape
    
    # https://stackoverflow.com/a/4744625
    input_aspect = float(width) / float(height)
    target_aspect = float(target_width) / float(target_height)

    if input_aspect > target_aspect:
        # Then crop the left and right edges:
        new_width = int(target_aspect * height)
        offset = (width - new_width) / 2
        box = (offset, 0, width - offset, height)
    else:
        # ... crop the top and bottom:
        new_height = int(width / target_aspect)
        offset = (height - new_height) / 2
        box = (0, offset, width, height - offset)

    # xyxy format, upper left and lower right corners
    box = (max(int(box[0]), 0),
           max(int(box[1]), 0),
           min(int(box[2]), width - 1),
           min(int(box[3]), height - 1))

    # Zoom in on the region of interest if needed
    cropped_mask = mask[box[1]:box[3], box[0]:box[2]]
    target_ratio = np.sum(cropped_mask) / 255 / (width * height)

    if target_ratio < 0.025 and mask.max() > 0:
        # Find bounding box of mask
        left = max(0, min(x for x in range(width) if mask[:, x].max() != 0))
        right = min(width - 1, max(x for x in range(width) if mask[:, x].max() != 0) + 1)
        top = max(0, min(y for y in range(height) if mask[y].max() != 0) - 1)
        bottom = min(height - 1, max(y for y in range(height) if mask[y].max() != 0) + 1)

        # Don't crop too tightly around the mask
        left = max(0, left - random.randint(0, (right - left) // 2))
        right = min(width - 1, right + random.randint(0, (right - left) // 2))
        top = max(0, top - random.randint(0, (bottom - top)))
        bottom = min(height - 1, bottom + random.randint(0, (bottom - top) // 2))

        # box should be at least as big as target size, put the target at some random position inside the target crop
        extra_hori = max(0, target_width - (right - left))
        extra_vert = max(0, target_height - (bottom - top))

        extra_left = random.randint(0, extra_hori)
        extra_top = random.randint(0, extra_vert)

        left -= extra_left
        right += (extra_hori - extra_left)

        top -= extra_top
        bottom += (extra_vert - extra_top)

        box = (left, top, right, bottom)

        # Fix box aspect ratio again
        box_width = box[2] - box[0]
        box_height = box[3] - box[1]
        box_aspect = float(box_width) / float(box_height)

        if box_aspect <= target_aspect:
            # Then expand the left and right edges:
            new_width = int(box_height * target_aspect)
            offset = (new_width - box_width) / 2
            box = (box[0] - offset, box[1], box[2] + offset, box[3])
        else:
            # ... expand the top and bottom:
            new_height = int(box_width / target_aspect)
            offset = (new_height - box_height) / 2
            box = (box[0], box[1] - offset, box[2], box[3] + offset)

    # xyxy format, upper left and lower right corners
    box = (max(int(box[0]), 0),
           max(int(box[1]), 0),
           min(int(box[2]), width - 1),
           min(int(box[3]), height - 1))

    return box

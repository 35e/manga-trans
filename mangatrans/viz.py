"""The annotated copy of a page written by ``--viz`` / ``--save-viz``.

Shows what each stage decided: the bubbles that were segmented, the fragments
the detector found, which group each ended up in, and what was thrown away and
why. That is the picture to look at when tuning anything.
"""

from __future__ import annotations

from pathlib import Path

from .pipeline import ART, BUBBLE, PLATE

REGION_TINT = (120, 225, 160)
KIND_COLOUR = {
    BUBBLE: (0, 150, 40),
    PLATE: (215, 130, 0),
    ART: (150, 60, 200),
}
FRAGMENT_COLOUR = (60, 130, 255)
DROPPED_COLOUR = (210, 40, 40)


def draw_visualisation(path: Path, page, out_path: Path, show_dropped: bool = True) -> None:
    import cv2  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    from PIL import Image, ImageDraw, ImageFont, ImageOps  # noqa: PLC0415

    image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    pixels = np.asarray(image).copy()

    used = {id(g.region) for g in page.groups if g.region is not None}
    tint = pixels.copy()
    for region in page.regions:
        if id(region) not in used:
            continue
        box = region.box
        tint[box.y0 : box.y1, box.x0 : box.x1][region.mask.astype(bool)] = REGION_TINT
    pixels = cv2.addWeighted(tint, 0.35, pixels, 0.65, 0)

    image = Image.fromarray(pixels)
    draw = ImageDraw.Draw(image)
    scale = max(1, min(image.size) // 400)
    try:
        font = ImageFont.load_default(size=13 * scale)
    except TypeError:  # Pillow < 10.1
        font = ImageFont.load_default()

    if show_dropped:
        for group in page.dropped:
            draw.rectangle(group.bbox.as_list(), outline=DROPPED_COLOUR, width=scale)

    for group in page.groups:
        for fragment in group.boxes:
            draw.rectangle(fragment.as_list(), outline=FRAGMENT_COLOUR, width=scale)

    for i, group in enumerate(page.groups, start=1):
        box = group.bbox
        colour = KIND_COLOUR.get(group.kind, DROPPED_COLOUR)
        draw.rectangle(box.as_list(), outline=colour, width=2 * scale)
        # A detector that reports confidence has it shown, because a page that
        # went wrong nearly always went wrong where the model was unsure, and
        # that is the box to look at first.
        label = f"{i}" if group.confidence >= 1.0 else f"{i} {group.confidence:.2f}"
        tx, ty = box.x0 + 2 * scale, max(0, box.y0 - 17 * scale)
        draw.rectangle(
            [tx - 2 * scale, ty, tx + 8 * scale * len(label), ty + 17 * scale],
            fill=colour,
        )
        draw.text((tx, ty), label, fill=(255, 255, 255), font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)

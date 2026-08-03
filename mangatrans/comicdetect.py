"""Text detection with comic-text-detector, a model trained on comics.

The model is dmMaze's `comic-text-detector
<https://github.com/dmMaze/comic-text-detector>`_, trained on ~13k anime and
comic pages (a third Manga109-s, a third DCM, a third synthetic). It answers in
one pass the three questions the heuristic pipeline needs a page of geometry and
a dozen thresholds to guess at:

``blk``
    a YOLOv5 head over *text blocks* - one box per utterance, with a confidence
    and a language. This is the grouping, decided by something that has seen
    thousands of pages, rather than by how far apart two fragments happen to be.
``seg``
    a per-pixel mask of the lettering, furigana and all. This is what the eraser
    wants, measured rather than inferred from the page's tones.
``det``
    a DBNet-style text line map. Unused here: the block boxes and the mask
    between them already say everything the pipeline asks for, and reading the
    line map needs a polygon-unclipping dependency for no gain.

It is distributed as ONNX and runs on ``cv2.dnn``, so this backend needs nothing
that is not already installed - no torch, no onnxruntime. Detection is also
fixed-cost: the page is letterboxed to 1024x1024 whatever its size, so the
memory budgeting that CRAFT needs (see :mod:`.detect`) does not apply here.
"""

from __future__ import annotations

from pathlib import Path

from .detectors import MODEL_NAME as MODEL_FILENAME
from .detectors import Detection, DetectionResult
from .geometry import Box

# The size the model was exported at. Not a tunable: the ONNX graph has it baked
# into its shapes.
INPUT_SIZE = 1024

# Below this a "block" is more likely a patch of texture the head fired on.
# Deliberately low - a doubtful block is worth reporting with its doubt attached
# rather than dropping, which is the failure the heuristic path could not see.
CONF_THRESHOLD = 0.4
NMS_THRESHOLD = 0.35
# Share of full confidence at which a pixel counts as lettering. The seg head is
# well separated, so this only has to avoid the anti-aliased rim.
MASK_THRESHOLD = 0.3

# Class index -> language, from the model's own label order.
LANGUAGES = {0: "eng", 1: "ja"}

# Two components of the mask belong to the same column of type when their
# extents along the cross-axis overlap by at least this share of the narrower
# one. Generous, because a kana and the kanji below it are set to the same width
# but rarely to the same pixel.
COLUMN_OVERLAP = 0.35
# A component smaller than this share of the block's biggest one is a tone dot,
# a speck of noise or a piece of furigana - none of which set the type size.
SPECK_SHARE = 0.18


def letterbox(image, size: int = INPUT_SIZE):
    """Scale ``image`` into a ``size`` square, padding bottom and right.

    The padding goes on two sides rather than four because that is what the
    model was exported with; centring it would shift every box by half the pad.
    Returns the letterboxed image and the padding, which is what maps a box back
    onto the original page.
    """
    import cv2  # noqa: PLC0415

    height, width = image.shape[:2]
    scale = min(size / height, size / width)
    new_w, new_h = int(round(width * scale)), int(round(height * scale))
    pad_w, pad_h = size - new_w, size - new_h
    if (width, height) != (new_w, new_h):
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    image = cv2.copyMakeBorder(
        image, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )
    return image, pad_w, pad_h


def decode_blocks(raw, conf_threshold: float, nms_threshold: float):
    """YOLO head -> ``(boxes xyxy, classes, confidences)`` in canvas pixels.

    ``raw`` is ``(N, 7)``: centre x, centre y, width, height, objectness, then
    one score per class. Boxes are already decoded to canvas coordinates by the
    exported graph, so all that is left is to score, threshold and suppress.
    """
    import cv2  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    empty = (np.zeros((0, 4), np.float32), np.zeros(0, int), np.zeros(0, np.float32))
    if raw.size == 0:
        return empty

    scores = raw[:, 5:]
    classes = scores.argmax(1)
    confidence = raw[:, 4] * scores.max(1)
    keep = confidence > conf_threshold
    raw, classes, confidence = raw[keep], classes[keep], confidence[keep]
    if not len(raw):
        return empty

    centre_x, centre_y, width, height = raw[:, 0], raw[:, 1], raw[:, 2], raw[:, 3]
    # cv2's NMS wants xywh from the top-left corner.
    boxes = np.stack(
        [centre_x - width / 2, centre_y - height / 2, width, height], axis=1
    )
    kept = cv2.dnn.NMSBoxes(
        boxes.tolist(), confidence.tolist(), conf_threshold, nms_threshold
    )
    if len(kept) == 0:
        return empty
    kept = np.asarray(kept).flatten()
    boxes, classes, confidence = boxes[kept], classes[kept], confidence[kept]
    xyxy = np.stack(
        [boxes[:, 0], boxes[:, 1], boxes[:, 0] + boxes[:, 2], boxes[:, 1] + boxes[:, 3]],
        axis=1,
    )
    return xyxy, classes, confidence


def columns_in(mask, box: Box, vertical: bool = True) -> list[Box]:
    """The columns (or lines) of type inside ``box``, from the lettering mask.

    The pipeline downstream measures the size of the type from its fragments -
    a fragment's short side is one character - so what it needs back is one box
    per column, which is what CRAFT used to hand it. The mask gives connected
    blobs of ink instead, so blobs that share a column are welded back together:
    two of them belong to the same column when they line up across it.

    Specks are dropped first. A dot screen that survived the mask, and furigana
    set beside a kanji, are both far smaller than the type they sit next to, and
    letting either into the median would report a type size half the truth.
    """
    import cv2  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    from .geometry import _UnionFind, union_box  # noqa: PLC0415

    patch = mask[box.y0 : box.y1, box.x0 : box.x1]
    if patch.size == 0:
        return []
    count, _, stats, _ = cv2.connectedComponentsWithStats(
        (patch > 0).astype(np.uint8), 8
    )
    blobs = [
        Box(
            int(stats[i, cv2.CC_STAT_LEFT]),
            int(stats[i, cv2.CC_STAT_TOP]),
            int(stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH]),
            int(stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT]),
        )
        for i in range(1, count)
    ]
    if not blobs:
        return []

    biggest = max(b.glyph_size for b in blobs)
    blobs = [b for b in blobs if b.glyph_size >= SPECK_SHARE * biggest]
    if not blobs:
        return []

    union = _UnionFind(len(blobs))
    for i in range(len(blobs)):
        for j in range(i + 1, len(blobs)):
            a, b = blobs[i], blobs[j]
            if vertical:
                overlap = min(a.x1, b.x1) - max(a.x0, b.x0)
                span = min(a.w, b.w)
            else:
                overlap = min(a.y1, b.y1) - max(a.y0, b.y0)
                span = min(a.h, b.h)
            if overlap > 0 and overlap >= COLUMN_OVERLAP * max(1, span):
                union.union(i, j)

    merged: dict[int, list[Box]] = {}
    for index, blob in enumerate(blobs):
        merged.setdefault(union.find(index), []).append(blob)

    columns = []
    for members in merged.values():
        local = union_box(members)
        # Back into page coordinates: the components were labelled in a crop.
        columns.append(
            Box(
                local.x0 + box.x0,
                local.y0 + box.y0,
                local.x1 + box.x0,
                local.y1 + box.y0,
            )
        )
    return columns


class ComicDetector:
    """comic-text-detector, run through OpenCV's ONNX backend."""

    name = "comic"

    def __init__(
        self,
        weights: Path,
        conf_threshold: float = CONF_THRESHOLD,
        nms_threshold: float = NMS_THRESHOLD,
        mask_threshold: float = MASK_THRESHOLD,
    ) -> None:
        import cv2  # noqa: PLC0415

        self.net = cv2.dnn.readNetFromONNX(str(weights))
        # Ask for the outputs by name: OpenCV has been known to return them in a
        # different order than the graph declares them, and the reference
        # implementation carries a workaround for exactly that.
        self.outputs = ("blk", "seg", "det")
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.mask_threshold = mask_threshold

    def __call__(self, image, log=lambda _m: None) -> DetectionResult:
        """Detect on one page. ``image`` is an RGB array."""
        import cv2  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415

        height, width = image.shape[:2]
        canvas, pad_w, pad_h = letterbox(image, INPUT_SIZE)
        blob = cv2.dnn.blobFromImage(
            canvas, scalefactor=1 / 255.0, size=(INPUT_SIZE, INPUT_SIZE)
        )
        self.net.setInput(blob)
        raw_blocks, raw_mask, _lines = self.net.forward(self.outputs)
        # Some OpenCV builds have handed these back in a different order than
        # they were asked for; the reference implementation carries a workaround
        # for it too. The three are unmistakable by shape - the YOLO head is the
        # only 3-dimensional one and the lettering mask the only single-channel
        # map - so check rather than trust, because the failure is silent: every
        # box would land somewhere plausible and be wrong.
        outputs = [raw_blocks, raw_mask, _lines]
        blocks = [o for o in outputs if o.ndim == 3]
        masks = [o for o in outputs if o.ndim == 4 and o.shape[1] == 1]
        if not blocks or not masks:
            raise SystemExit(
                "the detector model returned unexpected outputs "
                f"({[tuple(o.shape) for o in outputs]}); is "
                f"{MODEL_FILENAME} the right file?"
            )
        raw_blocks, raw_mask = blocks[0], masks[0]

        # The page occupies the canvas minus the padding, so this is what scales
        # a canvas coordinate back onto it.
        scale_x = width / (INPUT_SIZE - pad_w)
        scale_y = height / (INPUT_SIZE - pad_h)

        text_mask = (raw_mask[0, 0] > self.mask_threshold).astype(np.uint8) * 255
        text_mask = text_mask[: INPUT_SIZE - pad_h, : INPUT_SIZE - pad_w]
        text_mask = cv2.resize(
            text_mask, (width, height), interpolation=cv2.INTER_NEAREST
        )

        boxes, classes, confidences = decode_blocks(
            raw_blocks[0], self.conf_threshold, self.nms_threshold
        )
        blocks: list[Detection] = []
        fragments: list[Box] = []
        for xyxy, cls, conf in zip(boxes, classes, confidences):
            box = Box(
                int(round(xyxy[0] * scale_x)),
                int(round(xyxy[1] * scale_y)),
                int(round(xyxy[2] * scale_x)),
                int(round(xyxy[3] * scale_y)),
            ).clipped(width, height)
            if box.w <= 0 or box.h <= 0:
                continue
            language = LANGUAGES.get(int(cls), "unknown")
            # English is set in horizontal lines, Japanese in vertical columns;
            # for anything else the block's own shape is the better guess.
            vertical = box.h >= box.w if language != "eng" else False
            # A block whose lettering the mask missed still has to carry a
            # fragment, or there is nothing to measure the type size from.
            block_fragments = columns_in(text_mask, box, vertical=vertical) or [box]
            fragments.extend(block_fragments)
            blocks.append(
                Detection(
                    box=box,
                    confidence=float(conf),
                    language=language,
                    fragments=block_fragments,
                )
            )

        log(f"  {len(blocks)} text block(s), {len(fragments)} fragment(s)")
        return DetectionResult(
            fragments=fragments, blocks=blocks, text_mask=text_mask
        )

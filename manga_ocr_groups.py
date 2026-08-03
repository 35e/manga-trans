#!/usr/bin/env python3
"""OCR a manga page and return its text grouped per speech bubble.

Pipeline
--------
1. **Detect**    - comic-text-detector, a model trained on ~13k comic pages,
   returns the text *blocks* (one per utterance, each with a confidence and a
   language) and a per-pixel mask of the lettering. ``--detector craft`` instead
   detects fragments with EasyOCR's general-purpose CRAFT and reconstructs the
   blocks from geometry, which is what this did throughout before.
2. **Segment**   - the page is segmented into the *shapes* text sits on: a
   speech bubble is a bounded, compact, flat island of paper drawn around its
   lettering, and so is a caption box or a sign. Both polarities are found, so
   an inverted caption with white text on black works too. This no longer
   decides what *is* text - only what shape it is sitting on, which is what the
   eraser repaints and what the letterer fits English into.
3. **Recognise** - each group is cropped as a whole and passed to `manga-ocr`,
   which is trained on complete manga text blocks and handles vertical text,
   multiple lines and furigana on its own.
4. **Render**    - with ``--translate --render`` the original lettering is taken
   off - repainted in the bubble's own paper colour, inpainted only where it sat
   on artwork - and the translation is set into the largest rectangle that fits
   inside the bubble.

Nothing leaves the pipeline silently: whatever is found and then rejected lands
in the JSON under ``dropped``, with the reason that removed it.

``python -m mangatrans.evaluate --truth truth --pred out`` scores a run against
hand-corrected pages, so a change can be shown to be an improvement rather than
assumed to be one.

Usage
-----
    python manga_ocr_groups.py page.jpg
    python manga_ocr_groups.py page.jpg --json out.json --viz boxes.png
    python manga_ocr_groups.py page.jpg --text bubbles   # dialogue only
    python manga_ocr_groups.py page.jpg --no-ocr --viz boxes.png   # look fast

Arguments may also be folders - or left out entirely, in which case the current
folder is scanned. With ``--out-dir`` every page gets its own ``<name>.json``:

    python manga_ocr_groups.py pages --out-dir pages/out

The implementation lives in the ``mangatrans`` package; this file is the entry
point the container and ``run.sh`` invoke.
"""

from mangatrans.cli import main

if __name__ == "__main__":
    raise SystemExit(main())

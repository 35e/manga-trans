#!/usr/bin/env python3
"""OCR a manga page and return its text grouped per speech bubble.

Pipeline
--------
1. **Detect**    - EasyOCR's CRAFT detector finds the individual text fragments
   (roughly one box per column or line of glyphs). Its own line-merging is
   disabled so we keep the raw fragments.
2. **Segment**   - the page is segmented into the *shapes* text sits on: a
   speech bubble is a bounded, compact, flat island of paper drawn around its
   lettering, and so is a caption box or a sign. Both polarities are found, so
   an inverted caption with white text on black works too.
3. **Group**     - each fragment is matched to the shape it sits on, and the
   fragments of one shape become one text group. Grouping by proximity only
   happens *inside* a shape, so two bubbles can never be merged into one
   utterance. Text on no shape at all is judged by what is behind it: plain
   paper means narration or a title and is kept, artwork means a sound effect
   and is dropped unless ``--text all``.
4. **Recognise** - each group is cropped as a whole and passed to `manga-ocr`,
   which is trained on complete manga text blocks and handles vertical text,
   multiple lines and furigana on its own.
5. **Render**    - with ``--translate --render`` the original lettering is taken
   off - repainted in the bubble's own paper colour, inpainted only where it sat
   on artwork - and the translation is set into the largest rectangle that fits
   inside the bubble.

Detection runs at ``--canvas-size auto`` by default: the resolution is fitted to
the memory the machine actually has, so a large scan is downscaled for detection
instead of getting the process OOM-killed.

Usage
-----
    python manga_ocr_groups.py page.jpg
    python manga_ocr_groups.py page.jpg --json out.json --viz boxes.png
    python manga_ocr_groups.py page.jpg --text all       # keep sound effects too
    python manga_ocr_groups.py page.jpg --no-ocr --viz boxes.png   # tune fast

Arguments may also be folders - or left out entirely, in which case the current
folder is scanned. With ``--out-dir`` every page gets its own ``<name>.json``:

    python manga_ocr_groups.py pages --out-dir pages/out

The implementation lives in the ``mangatrans`` package; this file is the entry
point the container and ``run.sh`` invoke.
"""

from mangatrans.cli import main

if __name__ == "__main__":
    raise SystemExit(main())

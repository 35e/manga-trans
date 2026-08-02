"""manga-trans: OCR, translate and re-letter a manga page.

The pipeline is five stages, one module each:

1. :mod:`mangatrans.detect`   - CRAFT finds the raw text fragments.
2. :mod:`mangatrans.regions`  - the page is segmented into the shapes text sits
   on: speech bubbles, caption boxes, signs.
3. :mod:`mangatrans.pipeline` - fragments are matched to those shapes and
   gathered into one group per utterance; anything left over is judged on what
   is behind it and kept or dropped.
4. :mod:`mangatrans.erase` / :mod:`mangatrans.letter` - the original lettering
   is taken off and the translation set into the space it came out of.
5. :mod:`mangatrans.translate` - a local ollama model does the translating.
"""

from .geometry import Box, box_gap, group_boxes, sort_reading_order, union_box

__all__ = ["Box", "box_gap", "group_boxes", "sort_reading_order", "union_box"]

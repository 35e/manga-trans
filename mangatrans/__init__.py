"""manga-trans: OCR, translate and re-letter a manga page.

The pipeline is a handful of stages, one module each:

1. :mod:`mangatrans.detectors` - which detector finds the text, behind one
   interface. :mod:`mangatrans.comicdetect` runs comic-text-detector, a model
   trained on comics, which returns the text blocks and a mask of the lettering;
   :mod:`mangatrans.detect` runs EasyOCR's CRAFT, which returns fragments only.
2. :mod:`mangatrans.regions`  - the page is segmented into the shapes text sits
   on: speech bubbles, caption boxes, signs. This says what shape the text is
   sitting on, not whether it is text.
3. :mod:`mangatrans.pipeline` - one group per utterance, from the detector's own
   blocks where it has them and from proximity where it does not. Whatever is
   rejected is kept with the reason attached rather than discarded.
4. :mod:`mangatrans.erase` / :mod:`mangatrans.letter` - the original lettering
   is taken off and the translation set into the space it came out of.
5. :mod:`mangatrans.translate` - a local ollama model does the translating.
6. :mod:`mangatrans.evaluate`  - scores a run against hand-checked pages, so a
   change can be shown to be an improvement rather than assumed to be one.
"""

from .geometry import Box, box_gap, group_boxes, sort_reading_order, union_box

__all__ = ["Box", "box_gap", "group_boxes", "sort_reading_order", "union_box"]

"""Command line: discover images, run the pipeline, write what was asked for."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

from .comicdetect import CONF_THRESHOLD, MASK_THRESHOLD
from .detect import CANVAS_MIN
from .detectors import AUTO as DETECTOR_AUTO
from .detectors import DETECTORS, MODEL_ENV, MODEL_NAME
from .detectors import build as build_text_detector
from .erase import (
    AUTO,
    BLEED_GLYPHS,
    KNIT_GLYPHS,
    MODES,
    TIGHT,
    WIPE_GLYPHS,
    colour,
    erase_text,
    text_mask,
)
from .letter import render_page
from .pipeline import AUTO_TEXT, TEXT_CHOICES, Page, process_image
from .regions import CONTAINS, MAX_MIDTONE, MIN_SOLIDITY, SEAL_PX, page_masks
from .translate import OllamaError, translate_texts
from .viz import draw_visualisation

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".gif"}


# ---------------------------------------------------------------------------
# text output
# ---------------------------------------------------------------------------


def page_lines(page: Page, fmt: str = "both") -> list[str]:
    """The page's groups as text lines, in reading order."""
    lines = [f"# {page.path.name}"]
    for i, group in enumerate(page.groups, start=1):
        if fmt == "translation":
            if group.translation:
                lines.append(f"[{i}] {group.translation}")
        elif fmt == "original":
            if group.text:
                lines.append(f"[{i}] {group.text}")
        else:
            lines.append(f"[{i}] {group.text}")
            if group.translation:
                lines.append(f"    -> {group.translation}")
    return lines


def write_txt(pages: list[Page], out_path: Path, fmt: str = "both") -> None:
    blocks = ["\n".join(page_lines(page, fmt)) for page in pages]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# input discovery
# ---------------------------------------------------------------------------


def natural_key(path: Path) -> tuple:
    """Sort key that orders page2 before page10.

    Each chunk is a same-shaped tuple so that a numeric chunk never has to be
    compared against a text one ("001.webp" next to "image.png").
    """
    chunks = [
        (0, int(c), "") if c.isdigit() else (1, 0, c)
        for c in re.split(r"(\d+)", path.name.lower())
        if c
    ]
    return (path.parent.as_posix(), chunks)


def collect_images(
    paths: list[Path], recursive: bool = False, skip_dir: Path | None = None
) -> list[Path]:
    """Expand folders into the image files inside them.

    Files given explicitly are always kept, whatever their extension. Folders
    contribute the images they contain (``recursive`` also walks sub-folders);
    anything under ``skip_dir`` - the output folder - is left out so results are
    never fed back in as input.
    """
    found: list[Path] = []
    for path in paths:
        if path.is_dir():
            entries = path.rglob("*") if recursive else path.glob("*")
            found.extend(
                p for p in entries if p.is_file() and p.suffix.lower() in IMAGE_EXTS
            )
        elif path.is_file():
            found.append(path)
        else:
            raise SystemExit(f"no such file or folder: {path}")

    if skip_dir is not None and skip_dir.exists():
        skip = skip_dir.resolve()
        found = [p for p in found if skip not in p.resolve().parents]

    images: list[Path] = []
    seen: set[Path] = set()
    for path in sorted(found, key=natural_key):
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            images.append(path)
    return images


# ---------------------------------------------------------------------------
# arguments
# ---------------------------------------------------------------------------


def canvas_size_arg(value: str) -> int | None:
    """``--canvas-size``: ``auto`` (``None``, resolved per page) or a pixel count."""
    if value.strip().lower() == "auto":
        return None
    try:
        canvas = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected 'auto' or a number, got {value!r}")
    if canvas < CANVAS_MIN:
        raise argparse.ArgumentTypeError(f"must be at least {CANVAS_MIN}")
    return canvas


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="manga-trans",
        description="OCR a manga page, grouping text by speech bubble.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    default_input = Path(os.environ.get("MANGA_TRANS_INPUT", "."))
    default_out_dir = os.environ.get("MANGA_TRANS_OUT_DIR")

    parser.add_argument(
        "images",
        nargs="*",
        type=Path,
        default=[default_input],
        help="image file(s) or folder(s) to process; a folder is scanned for images",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="also scan sub-folders of the folders given",
    )

    what = parser.add_argument_group("what counts as text")
    what.add_argument(
        "--text",
        choices=TEXT_CHOICES,
        default=AUTO_TEXT,
        help="bubbles: only text inside a speech bubble, caption box or sign. "
        "page: also free-standing text on plain paper (narration, titles). "
        "all: also text painted over artwork (sound effects). "
        "auto: everything a detector that groups the page found (it already "
        "decided what is text), otherwise the same as page",
    )
    what.add_argument(
        "--all-text",
        dest="text",
        action="store_const",
        const="all",
        help="alias for --text all",
    )
    what.add_argument(
        "--plain-threshold",
        type=float,
        default=0.85,
        help="how plain the background behind free-standing text must be for it "
        "to count as text rather than a sound effect, 0-1",
    )
    what.add_argument(
        "--min-fragments",
        type=int,
        default=1,
        help="drop groups made of fewer detected fragments than this",
    )
    what.add_argument(
        "--min-group-px",
        type=int,
        default=12,
        help="drop groups smaller than this in both dimensions",
    )

    bubbles = parser.add_argument_group("bubble segmentation")
    bubbles.add_argument(
        "--min-solidity",
        type=float,
        default=MIN_SOLIDITY,
        help="how much of its convex hull a bubble must fill, 0-1 "
        "(lower to accept ragged shout balloons)",
    )
    bubbles.add_argument(
        "--max-midtone",
        type=float,
        default=MAX_MIDTONE,
        help="share of a bubble allowed to be neither paper nor ink "
        "(raise for toned or textured bubbles)",
    )
    bubbles.add_argument(
        "--seal",
        type=int,
        default=SEAL_PX,
        help="pixels of erosion used to seal anti-aliased gaps in bubble "
        "outlines before segmenting (0 disables)",
    )
    bubbles.add_argument(
        "--contains",
        type=float,
        default=CONTAINS,
        help="share of a text fragment that must sit on a bubble to belong to it",
    )
    bubbles.add_argument(
        "--max-bubble-ratio",
        type=float,
        default=10.0,
        help="how many times the size of its own text a bubble may be, before "
        "it is judged a blank patch of artwork instead",
    )
    bubbles.add_argument(
        "--no-inverted",
        action="store_true",
        help="do not look for inverted bubbles (white text on a dark plate)",
    )

    grouping = parser.add_argument_group("grouping")
    grouping.add_argument(
        "--gap",
        type=float,
        default=1.2,
        help="max gap between two neighbouring columns (or words) of one block "
        "of text, in multiples of the glyph size (higher = merge more)",
    )
    grouping.add_argument(
        "--stack-gap",
        type=float,
        default=3.0,
        help="same, for a fragment that continues the column (or line) another "
        "one started. Generous on purpose: a trailing '?' or an ellipsis is part "
        "of the sentence above it however far below it is set",
    )
    grouping.add_argument("--min-gap-px", type=float, default=0.0, help="gap floor in px")
    grouping.add_argument("--max-gap-px", type=float, default=None, help="gap ceiling in px")
    grouping.add_argument(
        "--order",
        choices=["rtl", "ltr", "none"],
        default="rtl",
        help="reading order of the returned groups (rtl = manga)",
    )

    which = parser.add_argument_group("which detector")
    which.add_argument(
        "--detector",
        choices=DETECTORS,
        default=DETECTOR_AUTO,
        help="comic: comic-text-detector, trained on manga - it finds the text "
        "blocks, so the bubble segmentation and grouping options below are only "
        "used to shape the lettering. craft: EasyOCR's general-purpose detector, "
        "with the blocks reconstructed by geometry. auto: comic if its weights "
        "are installed, else craft",
    )
    which.add_argument(
        "--detector-model",
        default=None,
        help=f"path to {MODEL_NAME} (default: ${MODEL_ENV}, then /opt/models "
        "and ~/.cache/manga-trans)",
    )
    which.add_argument(
        "--detector-conf",
        type=float,
        default=CONF_THRESHOLD,
        help="confidence a text block needs before the comic detector reports "
        "it at all",
    )
    which.add_argument(
        "--detector-mask-threshold",
        type=float,
        default=MASK_THRESHOLD,
        help="confidence a pixel needs to count as lettering, 0-1",
    )
    which.add_argument(
        "--min-confidence",
        type=float,
        default=0.0,
        help="drop text blocks the detector was less sure of than this. 0 keeps "
        "everything: a doubtful block is reported with its doubt attached, and "
        "the JSON records anything dropped and why",
    )

    detection = parser.add_argument_group("detection (EasyOCR/CRAFT)")
    detection.add_argument("--min-size", type=int, default=10)
    detection.add_argument(
        "--text-threshold",
        type=float,
        default=0.5,
        help="how sure the detector must be that a blob is a character. "
        "CRAFT's own default is 0.7, which is set for Latin prose; kana and "
        "kanji score lower per character and 0.5 finds about a fifth more of "
        "them without picking up artwork",
    )
    detection.add_argument(
        "--low-text",
        type=float,
        default=0.3,
        help="score at which a box stops growing. Lower makes the boxes hug "
        "faint strokes and thin punctuation instead of cutting them off",
    )
    detection.add_argument("--link-threshold", type=float, default=0.4)
    detection.add_argument(
        "--canvas-size",
        type=canvas_size_arg,
        default="auto",
        help="detection resolution in px (long side). 'auto' fits it to the "
        "memory this machine has, so large pages are not killed mid-run",
    )
    detection.add_argument(
        "--mag-ratio",
        type=float,
        default=1.0,
        help="magnify the page by this much before detecting. Raise it (1.5 is "
        "a good try) when small furigana is being missed; it costs about that "
        "much again in time and memory",
    )
    detection.add_argument("--detect-network", default="craft", choices=["craft", "dbnet18"])

    recognition = parser.add_argument_group("recognition (manga-ocr)")
    recognition.add_argument(
        "--model",
        default="kha-white/manga-ocr-base",
        help="HuggingFace model id or local path for manga-ocr",
    )
    recognition.add_argument(
        "--pad",
        type=float,
        default=0.15,
        help="padding added around a group before OCR, in multiples of glyph size",
    )
    recognition.add_argument(
        "--no-ocr",
        action="store_true",
        help="only detect and group (useful to tune without loading the OCR model)",
    )
    recognition.add_argument(
        "--keep-empty",
        dest="drop_empty",
        action="store_false",
        help="keep groups whose OCR result is empty",
    )
    recognition.add_argument("--cpu", action="store_true", help="force CPU inference")

    translation = parser.add_argument_group("translation (ollama)")
    translation.add_argument(
        "--translate",
        action="store_true",
        help="translate every text group with a local ollama model",
    )
    translation.add_argument(
        "--ollama-url",
        default=os.environ.get("OLLAMA_URL", "http://localhost:11434"),
        help="ollama server; inside a container use http://host.containers.internal:11434",
    )
    translation.add_argument(
        "--ollama-model",
        default=os.environ.get("OLLAMA_MODEL", "gemma4:12b"),
        help="model to translate with (`ollama list` shows what you have)",
    )
    translation.add_argument(
        "--target-lang", default="English", help="language to translate into"
    )
    translation.add_argument(
        "--ollama-timeout",
        type=float,
        default=180.0,
        help="seconds to wait for one ollama response",
    )

    render = parser.add_argument_group("rendering (--translate required)")
    render.add_argument(
        "--render",
        action="store_true",
        help="with --out-dir, write <name>.render.jpg: the page with the "
        "original text erased and the translation lettered in its place",
    )
    render.add_argument(
        "--render-to",
        type=Path,
        help="write the rendered page here instead (single image only)",
    )
    render.add_argument(
        "--font",
        help="TTF/OTF to letter with (default: DejaVu Sans Bold, then Pillow's own)",
    )
    render.add_argument(
        "--text-colour",
        "--text-color",
        dest="text_colour",
        default="black",
        help="colour of the lettering",
    )
    render.add_argument(
        "--halo-colour",
        "--halo-color",
        dest="halo_colour",
        default="white",
        help="outline drawn behind lettering that had to stay on artwork",
    )
    render.add_argument(
        "--line-spacing",
        type=float,
        default=0.16,
        help="extra gap between lines, in multiples of the font size",
    )
    render.add_argument(
        "--erase-pad",
        type=float,
        default=BLEED_GLYPHS,
        help="how far erasing spills past each glyph, in multiples of glyph size "
        "(raise if the original shows through)",
    )

    masking = parser.add_argument_group("masking (no translation needed)")
    masking.add_argument(
        "--clean",
        action="store_true",
        help="with --out-dir, write <name>.clean.png: the page with the Japanese "
        "covered over and nothing drawn in its place, ready to be lettered by hand",
    )
    masking.add_argument(
        "--clean-to", type=Path, help="write the cleaned page here (single image only)"
    )
    masking.add_argument(
        "--mask",
        action="store_true",
        help="with --out-dir, write <name>.mask.png: white where the lettering "
        "was, black everywhere else",
    )
    masking.add_argument(
        "--mask-to", type=Path, help="write the mask here (single image only)"
    )
    masking.add_argument(
        "--mask-mode",
        choices=list(MODES),
        default=None,
        help="how far the cover-up spreads. text: exactly over the lettering, "
        "which is the default for --clean/--mask. bubble: the whole inside of "
        "the bubble, which leaves the most room to letter English into. auto: "
        "measure the surface under the text and paint that back instead of a "
        "flat colour, which is the default for --render",
    )
    masking.add_argument(
        "--mask-colour",
        "--mask-color",
        dest="mask_colour",
        default="white",
        help="colour laid over the lettering in text/bubble mode",
    )
    masking.add_argument(
        "--mask-reach",
        type=float,
        default=WIPE_GLYPHS,
        help="how far past the detected text the cover-up looks for leftovers, "
        "in multiples of glyph size. Inside a bubble this is what picks up "
        "furigana and a leading ellipsis the detector never boxed; raise it if "
        "stray marks survive, lower it if a neighbouring utterance is caught",
    )
    masking.add_argument(
        "--mask-knit",
        type=float,
        default=KNIT_GLYPHS,
        help="how far the cover-up closes over the gaps between strokes and "
        "characters, in multiples of glyph size, so a column of type comes out "
        "as one patch rather than as separate glyph-shaped holes (0 traces "
        "each glyph exactly)",
    )

    output = parser.add_argument_group("output")
    output.add_argument(
        "--out-dir",
        type=Path,
        default=Path(default_out_dir) if default_out_dir else None,
        help="write one <name>.json per image into this folder "
        "(files already in it are never used as input)",
    )
    output.add_argument(
        "--save-viz",
        action="store_true",
        help="with --out-dir, also write an annotated <name>.boxes.png per image",
    )
    output.add_argument(
        "--txt",
        type=Path,
        help="write every page's text to one .txt file, in reading order "
        "(default with --out-dir: <out-dir>/pages.txt)",
    )
    output.add_argument(
        "--txt-format",
        choices=["both", "translation", "original"],
        default="both",
        help="what the .txt files contain",
    )
    output.add_argument(
        "--json", type=Path, help="write all pages to one JSON file ('-' for stdout)"
    )
    output.add_argument(
        "--viz", type=Path, help="write an annotated copy of the image (single image only)"
    )
    output.add_argument("--quiet", action="store_true", help="suppress progress logging")

    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def _load_masks(path: Path):
    """Page masks for rendering, recomputed from the file on disk."""
    import numpy as np  # noqa: PLC0415
    from PIL import Image, ImageOps  # noqa: PLC0415

    image = ImageOps.exif_transpose(Image.open(path)).convert("L")
    return page_masks(np.array(image))


def write_cover(path: Path, page, args, clean_to: Path | None, mask_to: Path | None):
    """Write the cleaned page and/or the mask of what it covered.

    Both come from one pass, because the mask is what the cleaning painted; a
    second pass could only disagree with the first. Each group's covered area
    lands back on the group, so the JSON says where there is now room to letter.
    """
    import cv2  # noqa: PLC0415
    from PIL import Image, ImageOps  # noqa: PLC0415

    import numpy as np  # noqa: PLC0415

    masks = _load_masks(path)
    mode = args.mask_mode or TIGHT
    image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")

    if mode == AUTO:
        # The surface under the lettering is measured rather than chosen, so the
        # page has to be painted to know what was covered.
        image = erase_text(
            image,
            page.groups,
            masks,
            args.erase_pad,
            mode=mode,
            reach_glyphs=args.mask_reach,
        )
        mask = (
            text_mask(
                masks,
                page.groups,
                masks.shape,
                bleed_glyphs=args.erase_pad,
                reach_glyphs=args.mask_reach,
                mode=mode,
            )
            if mask_to is not None
            else None
        )
    else:
        mask = text_mask(
            masks,
            page.groups,
            masks.shape,
            bleed_glyphs=args.erase_pad,
            knit_glyphs=args.mask_knit,
            reach_glyphs=args.mask_reach,
            mode=mode,
        )
        if clean_to is not None:
            pixels = np.asarray(image).copy()
            pixels[mask.astype(bool)] = colour(args.mask_colour)
            image = Image.fromarray(pixels)

    if clean_to is not None:
        clean_to.parent.mkdir(parents=True, exist_ok=True)
        image.save(clean_to)
    if mask_to is not None:
        mask_to.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(mask_to), mask)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    images = collect_images(args.images, args.recursive, skip_dir=args.out_dir)
    if not images:
        where = ", ".join(str(p) for p in args.images)
        raise SystemExit(f"no images found in {where}")
    if args.viz and len(images) > 1:
        raise SystemExit("--viz works with a single image at a time; use --save-viz")
    if args.render_to and len(images) > 1:
        raise SystemExit("--render-to takes a single image; use --render --out-dir")
    for flag, value in (("--clean-to", args.clean_to), ("--mask-to", args.mask_to)):
        if value and len(images) > 1:
            raise SystemExit(f"{flag} takes a single image; use --out-dir")
    if (args.clean or args.mask) and not args.out_dir:
        raise SystemExit(
            "--clean/--mask write into --out-dir; give one, or use --clean-to/--mask-to"
        )
    colour(args.mask_colour)  # fail now rather than after the models have loaded
    if args.translate and args.no_ocr:
        raise SystemExit("--translate needs the OCR text; drop --no-ocr")
    if (args.render or args.render_to) and not args.translate:
        raise SystemExit("--render draws the translation; add --translate")
    if args.render and not args.out_dir:
        raise SystemExit("--render writes into --out-dir; give one, or use --render-to")

    def log(msg: str) -> None:
        if not args.quiet:
            print(msg, file=sys.stderr)

    log(f"{len(images)} image(s) to process")

    log("loading text detector...")
    detector = build_text_detector(args.detector, args, log=log)
    log(f"  using the {detector.name} detector")

    mocr = None
    if not args.no_ocr:
        try:
            from manga_ocr import MangaOcr  # noqa: PLC0415
        except ImportError as exc:
            raise SystemExit(
                "manga-ocr is required for recognition: pip install manga-ocr"
            ) from exc
        log("loading manga-ocr...")
        mocr = MangaOcr(pretrained_model_name_or_path=args.model, force_cpu=args.cpu)

    if args.out_dir:
        args.out_dir.mkdir(parents=True, exist_ok=True)

    used_stems: set[str] = set()

    def out_stem(path: Path) -> str:
        """Unique output name, so same-named pages in different folders
        (--recursive) do not overwrite each other."""
        stem = path.stem
        if stem in used_stems:
            stem = f"{path.parent.name}_{stem}" if path.parent.name else stem
            suffix = 2
            while stem in used_stems:
                stem, suffix = f"{path.stem}_{suffix}", suffix + 1
        used_stems.add(stem)
        return stem

    pages: list[Page] = []
    for path in images:
        log(f"processing {path}...")
        page = process_image(path, detector, mocr, args, log=log)

        if args.translate and page.groups:
            log(f"translating {len(page.groups)} group(s) with {args.ollama_model}...")
            try:
                translations = translate_texts(
                    [g.text for g in page.groups],
                    url=args.ollama_url,
                    model=args.ollama_model,
                    target_lang=args.target_lang,
                    timeout=args.ollama_timeout,
                    log=log,
                )
            except OllamaError as exc:
                raise SystemExit(f"translation failed: {exc}") from exc
            for group, translated in zip(page.groups, translations):
                group.translation = translated

        pages.append(page)

        print(f"=== {path} - {len(page.groups)} text group(s) ===")
        for i, group in enumerate(page.groups, start=1):
            box = group.bbox
            print(
                f"[{i}] {group.kind} bbox=({box.x0},{box.y0})-({box.x1},{box.y1}) "
                f"fragments={len(group.boxes)}"
            )
            if group.text:
                print(f"    {group.text}")
            if group.translation:
                print(f"    -> {group.translation}")
        print()

        if args.out_dir:
            stem = out_stem(path)

            # Before the JSON: covering the page is what fills in each group's
            # mask_bbox, and that belongs in the file with the rest of it.
            clean_path = args.out_dir / f"{stem}.clean.png" if args.clean else None
            mask_path = args.out_dir / f"{stem}.mask.png" if args.mask else None
            if clean_path or mask_path:
                write_cover(path, page, args, clean_path, mask_path)
                for written in (clean_path, mask_path):
                    if written:
                        log(f"wrote {written}")

            page_json = args.out_dir / f"{stem}.json"
            page_json.write_text(
                json.dumps(page.to_dict(), ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            log(f"wrote {page_json}")

            page_txt = args.out_dir / f"{stem}.txt"
            write_txt([page], page_txt, args.txt_format)
            log(f"wrote {page_txt}")

            if args.save_viz:
                viz_path = args.out_dir / f"{stem}.boxes.png"
                draw_visualisation(path, page, viz_path)
                log(f"wrote {viz_path}")

            if args.render:
                render_path = args.out_dir / f"{stem}.render.jpg"
                count = render_page(path, page, render_path, _load_masks(path), args)
                log(f"wrote {render_path} ({count} bubble(s) lettered)")

    if args.clean_to or args.mask_to:
        write_cover(images[0], pages[0], args, args.clean_to, args.mask_to)
        for written in (args.clean_to, args.mask_to):
            if written:
                log(f"wrote {written}")

    if args.viz:
        draw_visualisation(images[0], pages[0], args.viz)
        log(f"wrote {args.viz}")

    if args.render_to:
        count = render_page(
            images[0], pages[0], args.render_to, _load_masks(images[0]), args
        )
        log(f"wrote {args.render_to} ({count} bubble(s) lettered)")

    # All pages in one file, in the order they were processed.
    txt_path = args.txt or (args.out_dir / "pages.txt" if args.out_dir else None)
    if txt_path:
        write_txt(pages, txt_path, args.txt_format)
        log(f"wrote {txt_path}")

    if args.json:
        payload = {"pages": [p.to_dict() for p in pages]}
        text = json.dumps(payload, ensure_ascii=False, indent=2)
        if str(args.json) == "-":
            print(text)
        else:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            args.json.write_text(text + "\n", encoding="utf-8")
            log(f"wrote {args.json}")

    return 0

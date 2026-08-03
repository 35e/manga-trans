"""Scoring a run against pages that have been checked by hand.

Every knob in this project trades one kind of mistake for another, and with
enough of them you can spend a long afternoon making a page better and a volume
worse. Judging by eye cannot tell those apart: four pages will always look
roughly right, and the failure that matters - a bubble that quietly went missing
on page fifty - is invisible precisely because there is nothing there to see.

So: score the run. Ground truth is a folder of the same JSON the pipeline
already writes, with the boxes and the text corrected by hand, which means
building it costs a run and an hour rather than a labelling project:

    ./run.sh pages --out-dir truth        # then correct truth/*.json by hand
    ./run.sh pages --out-dir out
    ./run.sh eval --truth truth --pred out

Two numbers come out, and they answer different questions. **Detection** F1 says
whether the text was found and cut up into the right utterances; a change to the
detector or the grouping moves this. **CER** - character error rate over the
boxes that did match - says whether what was found was read correctly; a change
to the recognition model moves this. Reporting them apart is the point, because
a change that finds three more bubbles and reads them badly is not an
improvement and a single number would call it one.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

from .geometry import Box

# Share of the union two boxes must have in common to count as the same bubble.
# 0.5 is the usual convention and is forgiving of a few pixels of padding either
# way, which is all the disagreement a correct detection ever shows.
IOU_THRESHOLD = 0.5


def iou(a: Box, b: Box) -> float:
    """Intersection over union of two boxes, 0-1."""
    inter = a.intersection(b)
    if inter.w <= 0 or inter.h <= 0:
        return 0.0
    overlap = inter.area
    union = a.area + b.area - overlap
    return overlap / union if union else 0.0


def edit_distance(hypothesis: str, reference: str) -> int:
    """Levenshtein distance, in characters."""
    if hypothesis == reference:
        return 0
    if not hypothesis:
        return len(reference)
    if not reference:
        return len(hypothesis)
    previous = list(range(len(reference) + 1))
    for i, h in enumerate(hypothesis, start=1):
        current = [i]
        for j, r in enumerate(reference, start=1):
            current.append(
                min(
                    previous[j] + 1,  # deletion
                    current[j - 1] + 1,  # insertion
                    previous[j - 1] + (h != r),  # substitution
                )
            )
        previous = current
    return previous[-1]


def match(
    predicted: list[Box], truth: list[Box], threshold: float = IOU_THRESHOLD
) -> list[tuple[int, int]]:
    """Pair predicted boxes with true ones, best overlap first.

    Greedy rather than optimal: with boxes as well separated as speech bubbles
    the two agree, and a greedy pass is easier to reason about when a score
    looks wrong.
    """
    scored = [
        (iou(p, t), pi, ti)
        for pi, p in enumerate(predicted)
        for ti, t in enumerate(truth)
        if iou(p, t) >= threshold
    ]
    scored.sort(reverse=True)

    pairs: list[tuple[int, int]] = []
    used_pred: set[int] = set()
    used_truth: set[int] = set()
    for _score, pi, ti in scored:
        if pi in used_pred or ti in used_truth:
            continue
        used_pred.add(pi)
        used_truth.add(ti)
        pairs.append((pi, ti))
    return pairs


@dataclass
class Score:
    """Detection counts and recognition errors, for a page or a whole run."""

    pages: int = 0
    matched: int = 0  # found, and where the truth said
    spurious: int = 0  # found, but nothing there (false positive)
    missed: int = 0  # in the truth, never found (false negative)
    edits: int = 0  # character edits over matched pairs
    reference_chars: int = 0
    exact: int = 0  # matched pairs read perfectly
    scored_pairs: int = 0  # matched pairs that had truth text to compare
    worst: list[tuple[float, str, str, str]] = field(default_factory=list)

    def add(self, other: "Score") -> None:
        self.pages += other.pages
        self.matched += other.matched
        self.spurious += other.spurious
        self.missed += other.missed
        self.edits += other.edits
        self.reference_chars += other.reference_chars
        self.exact += other.exact
        self.scored_pairs += other.scored_pairs
        self.worst += other.worst

    @property
    def precision(self) -> float:
        found = self.matched + self.spurious
        return self.matched / found if found else 0.0

    @property
    def recall(self) -> float:
        expected = self.matched + self.missed
        return self.matched / expected if expected else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0

    @property
    def cer(self) -> float:
        """Character error rate over matched pairs. Lower is better; 0 is perfect.

        Micro-averaged - total edits over total reference characters - so a long
        line of dialogue counts for more than a one-character grunt, which is
        what you want when the question is "how much of this page is wrong?".
        """
        return self.edits / self.reference_chars if self.reference_chars else 0.0

    @property
    def exact_rate(self) -> float:
        return self.exact / self.scored_pairs if self.scored_pairs else 0.0


def _groups(page: dict) -> list[dict]:
    return page.get("groups", [])


def load_pages(path: Path) -> dict[str, dict]:
    """Read one JSON file, or every JSON in a folder, keyed by image name.

    Accepts either shape the pipeline writes: the combined ``{"pages": [...]}``
    from ``--json``, or the one-file-per-page form from ``--out-dir``.
    """
    files = sorted(path.glob("*.json")) if path.is_dir() else [path]
    if not files:
        raise SystemExit(f"no JSON found in {path}")

    pages: dict[str, dict] = {}
    for file in files:
        try:
            data = json.loads(file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SystemExit(f"cannot read {file}: {exc}") from exc
        for page in data.get("pages", [data]):
            if "image" not in page:
                continue
            pages[Path(page["image"]).name] = page
    return pages


def score_page(
    predicted: dict, truth: dict, threshold: float = IOU_THRESHOLD, name: str = ""
) -> Score:
    pred_groups, truth_groups = _groups(predicted), _groups(truth)
    pred_boxes = [Box(*g["bbox"]) for g in pred_groups]
    truth_boxes = [Box(*g["bbox"]) for g in truth_groups]
    pairs = match(pred_boxes, truth_boxes, threshold)

    score = Score(
        pages=1,
        matched=len(pairs),
        spurious=len(pred_boxes) - len(pairs),
        missed=len(truth_boxes) - len(pairs),
    )
    for pi, ti in pairs:
        reference = (truth_groups[ti].get("text") or "").strip()
        hypothesis = (pred_groups[pi].get("text") or "").strip()
        if not reference:
            continue  # nothing to compare against; --no-ocr truth, most likely
        distance = edit_distance(hypothesis, reference)
        score.edits += distance
        score.reference_chars += len(reference)
        score.scored_pairs += 1
        if distance == 0:
            score.exact += 1
        else:
            score.worst.append((distance / len(reference), name, reference, hypothesis))
    return score


def score_run(
    predicted: dict[str, dict], truth: dict[str, dict], threshold: float = IOU_THRESHOLD
) -> tuple[Score, list[tuple[str, Score]]]:
    total = Score()
    per_page: list[tuple[str, Score]] = []
    for name in sorted(truth):
        if name not in predicted:
            print(f"warning: no prediction for {name}, skipping", file=sys.stderr)
            continue
        page = score_page(predicted[name], truth[name], threshold, name=name)
        per_page.append((name, page))
        total.add(page)
    return total, per_page


def report(total: Score, per_page: list[tuple[str, Score]], show_worst: int = 5) -> str:
    lines = [
        f"{'page':<28} {'found':>6} {'miss':>5} {'spur':>5} "
        f"{'recall':>7} {'prec':>6} {'F1':>6} {'CER':>7}",
        "-" * 74,
    ]
    for name, page in per_page:
        lines.append(
            f"{name[:28]:<28} {page.matched:>6} {page.missed:>5} {page.spurious:>5} "
            f"{page.recall:>7.1%} {page.precision:>6.1%} {page.f1:>6.1%} "
            f"{page.cer:>7.1%}"
        )
    lines += [
        "-" * 74,
        f"{'TOTAL (' + str(total.pages) + ' pages)':<28} "
        f"{total.matched:>6} {total.missed:>5} {total.spurious:>5} "
        f"{total.recall:>7.1%} {total.precision:>6.1%} {total.f1:>6.1%} "
        f"{total.cer:>7.1%}",
        "",
        f"detection   recall {total.recall:.1%}  precision {total.precision:.1%}  "
        f"F1 {total.f1:.1%}   ({total.missed} missed, {total.spurious} spurious)",
    ]
    if total.scored_pairs:
        lines.append(
            f"recognition CER {total.cer:.1%}  read perfectly "
            f"{total.exact}/{total.scored_pairs} ({total.exact_rate:.1%})"
        )
    else:
        lines.append(
            "recognition not scored: the ground truth has no text in it "
            "(was it built with --no-ocr?)"
        )

    if show_worst and total.worst:
        lines += ["", f"worst {show_worst} reads:"]
        for rate, name, reference, hypothesis in sorted(total.worst, reverse=True)[
            :show_worst
        ]:
            lines.append(f"  {rate:>6.0%}  {name}")
            lines.append(f"          want: {reference}")
            lines.append(f"          got : {hypothesis or '(nothing)'}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="manga-trans eval",
        description="Score a run against hand-checked pages.",
    )
    parser.add_argument(
        "--truth",
        type=Path,
        required=True,
        help="hand-corrected JSON: one file, or a folder of them",
    )
    parser.add_argument(
        "--pred",
        type=Path,
        required=True,
        help="the run to score, in the same shape",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=IOU_THRESHOLD,
        help="overlap at which a predicted box counts as the true one",
    )
    parser.add_argument(
        "--worst",
        type=int,
        default=5,
        help="how many of the worst-read bubbles to print (0 for none)",
    )
    parser.add_argument(
        "--json", action="store_true", help="print the totals as JSON instead"
    )
    args = parser.parse_args(argv)

    truth = load_pages(args.truth)
    predicted = load_pages(args.pred)
    if not truth:
        raise SystemExit(f"no pages found in {args.truth}")

    total, per_page = score_run(predicted, truth, args.iou)
    if args.json:
        print(
            json.dumps(
                {
                    "pages": total.pages,
                    "matched": total.matched,
                    "missed": total.missed,
                    "spurious": total.spurious,
                    "recall": round(total.recall, 4),
                    "precision": round(total.precision, 4),
                    "f1": round(total.f1, 4),
                    "cer": round(total.cer, 4),
                    "exact_rate": round(total.exact_rate, 4),
                },
                indent=2,
            )
        )
    else:
        print(report(total, per_page, args.worst))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

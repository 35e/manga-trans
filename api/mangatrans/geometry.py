from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Box:
    """Axis-aligned box in image pixels, x1/y1 exclusive."""

    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def w(self) -> int:
        return self.x1 - self.x0

    @property
    def h(self) -> int:
        return self.y1 - self.y0

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2

    def clipped(self, width: int, height: int) -> Box:
        return Box(
            min(max(0, self.x0), width),
            min(max(0, self.y0), height),
            min(max(0, self.x1), width),
            min(max(0, self.y1), height),
        )

    def moved(self, dx: int, dy: int) -> Box:
        return Box(self.x0 + dx, self.y0 + dy, self.x1 + dx, self.y1 + dy)

    def as_list(self) -> list[int]:
        return [self.x0, self.y0, self.x1, self.y1]

    @classmethod
    def from_list(cls, values) -> Box:
        """One [x0, y0, x1, y1], with the corners put in order."""
        x0, y0, x1, y1 = (round(float(v)) for v in values)
        return cls(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))

from dataclasses import dataclass

@dataclass
class ECPoint:
    x: int | None
    y: int | None

    @property
    def is_infinity(self) -> bool:
        return self.x is None and self.y is None

@dataclass
class EllipticCurve:
    p: int
    a: int
    b: int
    g: ECPoint
    n: int

    def is_on_curve(self, point: ECPoint) -> bool:
        if point.is_infinity:
            return True
        x, y = point.x, point.y
        if x is None or y is None:
            return False
        lhs = (y * y) % self.p
        rhs = (x * x * x + self.a * x + self.b) % self.p
        return lhs == rhs
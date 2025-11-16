
from constants import POINT_INFINITY
from dataclass import ECPoint, EllipticCurve


def secp256k1() -> EllipticCurve:
    p = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
    a = 0
    b = 7
    g = ECPoint(
        55066263022277343669578718895168534326250603453777594175500187360389116729240,
        32670510020758816978083085130507043184471273380659243275938904335757337482424,
    )
    n = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
    return EllipticCurve(p=p, a=a, b=b, g=g, n=n)

def p256() -> EllipticCurve:
    a=-3
    b=41058363725152142129326129780047268409114441015993725554835256314039467401291
    p=0xffffffff00000001000000000000000000000000ffffffffffffffffffffffff
    g = ECPoint(
        0x6b17d1f2e12c4247f8bce6e563a440f277037d812deb33a0f4a13945d898c296,
        0x4fe342e2fe1a7f9b8ee7eb4a7c0f9e162bce33576b315ececbb6406837bf51f5
    )
    n = 0xffffffff00000000ffffffffffffffffbce6faada7179e84f3b9cac2fc632551
    return EllipticCurve(p=p, a=a, b=b, g=g, n=n)

def inverse_mod(k: int, p: int) -> int:
    if k == 0:
        raise ZeroDivisionError("Cannot invert zero modulo prime")
    return pow(k, -1, p)


def point_neg(point: ECPoint, curve: EllipticCurve) -> ECPoint:
    if point.is_infinity:
        return POINT_INFINITY
    assert point.x is not None and point.y is not None
    return ECPoint(point.x, (-point.y) % curve.p)

def point_add(p1: ECPoint, p2: ECPoint, curve: EllipticCurve) -> ECPoint:
    if p1.is_infinity:
        return p2
    if p2.is_infinity:
        return p1
    assert p1.x is not None and p1.y is not None
    assert p2.x is not None and p2.y is not None

    if p1.x == p2.x and (p1.y != p2.y or p1.y == 0):
        return POINT_INFINITY
    
    if p1.x == p2.x:
        numerator = (3 * p1.x * p1.x + curve.a) % curve.p
        denominator = (2 * p1.y) % curve.p
        slope = numerator * inverse_mod(denominator, curve.p)
    else:
        numerator = (p2.y - p1.y) % curve.p
        denominator = (p2.x - p1.x) % curve.p
        slope = numerator * inverse_mod(denominator, curve.p)

    slope %= curve.p
    x3 = (slope * slope - p1.x - p2.x) % curve.p
    y3 = (slope * (p1.x - x3) - p1.y) % curve.p
    return ECPoint(x3, y3)


def scalar_mult(k: int, point: ECPoint, curve: EllipticCurve) -> ECPoint:
    if k < 0:
        return scalar_mult(-k, point_neg(point, curve), curve)
    if point.is_infinity:
        return POINT_INFINITY

    result = POINT_INFINITY
    addend = point

    while k:
        if k & 1:
            result = point_add(result, addend, curve)
        addend = point_add(addend, addend, curve)
        k >>= 1

    return result
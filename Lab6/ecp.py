"""Minimal GOST R 34.10-2012 CLI without auxiliary examples.

This script focuses on signing and verifying messages using explicit
hexadecimal parameters (message, private key, and public key).  It relies on
the Streebog hash implementation provided by :mod:`main`.
"""

from __future__ import annotations

import argparse
import secrets
from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Tuple

from gost34_11 import streebog


# --- Elliptic curve primitives -------------------------------------------------


def _mod_inv(value: int, modulus: int) -> int:
    value %= modulus
    if value == 0:
        raise ZeroDivisionError("inverse of zero modulo p is undefined")
    return pow(value, -1, modulus)


@dataclass(frozen=True)
class CurveParameters:
    p: int
    a: int
    b: int
    q: int
    x_p: int
    y_p: int

    @property
    def base_point(self) -> "Point":
        return Point(self.x_p, self.y_p, self)


class Point:
    __slots__ = ("x", "y", "curve")

    def __init__(self, x: int, y: int, curve: CurveParameters) -> None:
        self.x = x % curve.p
        self.y = y % curve.p
        self.curve = curve


PointOrInfinity = Optional[Point]


def _point_add(p1: PointOrInfinity, p2: PointOrInfinity) -> PointOrInfinity:
    if p1 is None:
        return p2
    if p2 is None:
        return p1

    curve = p1.curve
    if curve is not p2.curve:
        raise ValueError("Points belong to different curves")

    if p1.x == p2.x and (p1.y + p2.y) % curve.p == 0:
        return None

    if p1.x == p2.x and p1.y == p2.y:
        numerator = (3 * p1.x * p1.x + curve.a) % curve.p
        denominator = (2 * p1.y) % curve.p
    else:
        numerator = (p2.y - p1.y) % curve.p
        denominator = (p2.x - p1.x) % curve.p

    slope = (numerator * _mod_inv(denominator, curve.p)) % curve.p
    x3 = (slope * slope - p1.x - p2.x) % curve.p
    y3 = (slope * (p1.x - x3) - p1.y) % curve.p
    return Point(x3, y3, curve)


def _scalar_mul(k: int, point: PointOrInfinity) -> PointOrInfinity:
    if point is None:
        return None

    result: PointOrInfinity = None
    addend: PointOrInfinity = point
    k %= point.curve.q

    while k:
        if k & 1:
            result = _point_add(result, addend)
        addend = _point_add(addend, addend)
        k >>= 1
    return result


# --- Signature routines -------------------------------------------------------


def _hash_to_int(message: bytes, curve: CurveParameters, digest_size: int) -> int:
    digest = streebog(message, digest_size)
    e = int.from_bytes(digest, "big") % curve.q
    return e or 1


def sign(
    message: bytes,
    private_key: int,
    curve: CurveParameters,
    *,
    digest_size: int = 512,
    randfunc: Optional[Callable[[int], int]] = None,
) -> Tuple[int, int]:
    if not (0 < private_key < curve.q):
        raise ValueError("private_key must satisfy 0 < d < q")

    e = _hash_to_int(message, curve, digest_size)
    rand = randfunc or (lambda upper: secrets.randbelow(upper - 1) + 1)

    while True:
        k = rand(curve.q)
        if not (0 < k < curve.q):
            continue
        c = _scalar_mul(k, curve.base_point)
        if c is None:
            continue
        r = c.x % curve.q
        if r == 0:
            continue
        s = (r * private_key + k * e) % curve.q
        if s == 0:
            continue
        return r, s


def verify(
    message: bytes,
    signature: Tuple[int, int],
    public_key: PointOrInfinity,
    curve: CurveParameters,
    *,
    digest_size: int = 512,
) -> bool:
    r, s = signature
    if not (0 < r < curve.q and 0 < s < curve.q):
        return False
    if public_key is None:
        return False

    e = _hash_to_int(message, curve, digest_size)
    try:
        v = _mod_inv(e, curve.q)
    except ZeroDivisionError:
        return False

    z1 = (s * v) % curve.q
    z2 = (-r * v) % curve.q
    c = _point_add(_scalar_mul(z1, curve.base_point), _scalar_mul(z2, public_key))
    if c is None:
        return False
    return c.x % curve.q == r


# --- RFC test parameters ------------------------------------------------------


RFC_TEST_CURVE = CurveParameters(
    p=int(
        "57896044618658097711785492504343953926634992332820282019728792003956564821041"
    ),
    a=7,
    b=int(
        "43308876546767276905765904595650931995942111794451039583252968842033849580414"
    ),
    q=int(
        "57896044618658097711785492504343953927082934583725450622380973592137631069619"
    ),
    x_p=2,
    y_p=int(
        "4018974056539037503335449422937059775635739389905545080690979365213431566280"
    ),
)


# --- CLI helpers --------------------------------------------------------------


def _parse_int(value: str) -> int:
    text = value.strip().lower()
    if text.startswith("0x"):
        return int(text, 16)
    try:
        return int(text, 10)
    except ValueError as exc:
        raise SystemExit(f"Unable to parse integer: {value}") from exc


def _parse_hex_bytes(value: str) -> bytes:
    text = "".join(value.split())
    if len(text) % 2 != 0:
        text = "0" + text
    try:
        return bytes.fromhex(text)
    except ValueError as exc:
        raise SystemExit("Message must be provided as hexadecimal text") from exc


def _parse_signature_hex(value: str, curve: CurveParameters) -> Tuple[int, int]:
    text = "".join(value.split())
    width = max(1, (curve.q.bit_length() + 7) // 8) * 2
    if len(text) != width * 2:
        raise SystemExit(
            f"Signature hex length must be {width * 2} characters for this curve"
        )
    r = int(text[:width], 16)
    s = int(text[width:], 16)
    return r, s


def _make_public_key(curve: CurveParameters, x_value: str, y_value: str) -> Point:
    return Point(_parse_int(x_value), _parse_int(y_value), curve)


def _rand_from_hex(value: Optional[str]) -> Optional[Callable[[int], int]]:
    if value is None:
        return None
    scalar = _parse_int(value)
    return lambda upper: scalar % upper or upper - 1


# --- Command handlers ---------------------------------------------------------


def _cmd_sign(args: argparse.Namespace) -> None:
    message = _parse_hex_bytes(args.message_hex)
    private_key = _parse_int(args.private_key)
    randfunc = _rand_from_hex(args.random)
    r, s = sign(
        message,
        private_key,
        RFC_TEST_CURVE,
        digest_size=args.digest,
        randfunc=randfunc,
    )
    width = max(1, (RFC_TEST_CURVE.q.bit_length() + 7) // 8)
    signature_hex = (r.to_bytes(width, "big") + s.to_bytes(width, "big")).hex()
    print(signature_hex)


def _cmd_verify(args: argparse.Namespace) -> None:
    message = _parse_hex_bytes(args.message_hex)
    if args.signature:
        signature = _parse_signature_hex(args.signature, RFC_TEST_CURVE)
    else:
        if not args.signature_r or not args.signature_s:
            raise SystemExit("Provide --signature or both --signature-r and --signature-s")
        signature = (_parse_int(args.signature_r), _parse_int(args.signature_s))

    public_key = _make_public_key(RFC_TEST_CURVE, args.public_key_x, args.public_key_y)
    valid = verify(
        message,
        signature,
        public_key,
        RFC_TEST_CURVE,
        digest_size=args.digest,
    )
    print("VALID" if valid else "INVALID")


# --- CLI ----------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Minimal GOST R 34.10-2012 signer/verifier (hex-based)."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    sign_parser = subparsers.add_parser("sign", help="Generate a signature.")
    sign_parser.add_argument("--message-hex", required=True, help="Message bytes in hex.")
    sign_parser.add_argument("--private-key", required=True, help="Private key d (hex or decimal).")
    sign_parser.add_argument(
        "--random",
        help="Optional deterministic scalar k for testing (hex or decimal).",
    )
    sign_parser.add_argument(
        "--digest",
        type=int,
        choices=(256, 512),
        default=512,
        help="Digest size in bits (default: 512).",
    )
    sign_parser.set_defaults(func=_cmd_sign)

    verify_parser = subparsers.add_parser("verify", help="Check a signature.")
    verify_parser.add_argument("--message-hex", required=True, help="Message bytes in hex.")
    verify_parser.add_argument("--public-key-x", required=True, help="Public key x-coordinate (hex or decimal).")
    verify_parser.add_argument("--public-key-y", required=True, help="Public key y-coordinate (hex or decimal).")
    verify_parser.add_argument("--signature", help="Concatenated r||s signature hex.")
    # verify_parser.add_argument("--signature-r", help="Signature component r (hex or decimal).")
    # verify_parser.add_argument("--signature-s", help="Signature component s (hex or decimal).")
    verify_parser.add_argument(
        "--digest",
        type=int,
        choices=(256, 512),
        default=512,
        help="Digest size in bits (default: 512).",
    )
    verify_parser.set_defaults(func=_cmd_verify)

    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

from __future__ import annotations

import argparse
import secrets
from constants import ENCODING_K, METADATA_BITS, METADATA_MASK, POINT_INFINITY, Ciphertext
from curve import p256, point_add, point_neg, scalar_mult, secp256k1
from dataclass import ECPoint, EllipticCurve

def tonelli_shanks(n: int, p: int) -> int | None:
    if n == 0:
        return 0
    if pow(n, (p - 1) // 2, p) != 1:
        return None
    if p % 4 == 3:
        return pow(n, (p + 1) // 4, p)

    q = p - 1
    s = 0
    while q % 2 == 0:
        q //= 2
        s += 1

    z = 2
    while pow(z, (p - 1) // 2, p) != p - 1:
        z += 1

    m = s
    c = pow(z, q, p)
    t = pow(n, q, p)
    r = pow(n, (q + 1) // 2, p)

    while t != 1:
        temp = t
        i = 0
        while temp != 1:
            temp = pow(temp, 2, p)
            i += 1
            if i == m:
                return None

        b = pow(c, 1 << (m - i - 1), p)
        m = i
        c = pow(b, 2, p)
        t = (t * c) % p
        r = (r * b) % p

    return r

def encode_message(message: bytes, curve: EllipticCurve, k: int = ENCODING_K) -> ECPoint:
    if len(message) > METADATA_MASK:
        raise ValueError(f"Message chunk must be at most {METADATA_MASK} bytes")

    msg_int = int.from_bytes(message, "big") if message else 0
    base_x = msg_int * k

    for j in range(METADATA_MASK + 1):
        metadata = (len(message) << METADATA_BITS) | j
        candidate_x = base_x + metadata
        if candidate_x >= curve.p:
            break
        rhs = (candidate_x * candidate_x * candidate_x + curve.a * candidate_x + curve.b) % curve.p
        y = tonelli_shanks(rhs, curve.p)
        if y is not None:
            if y % 2 != 0:
                y = curve.p - y
            point = ECPoint(candidate_x, y)
            if curve.is_on_curve(point):
                return point

    raise ValueError("Failed to encode message as curve point")


def decode_message(point: ECPoint, k: int = ENCODING_K) -> bytes:
    if point.is_infinity or point.x is None:
        raise ValueError("Cannot decode point at infinity")
    metadata = point.x % k
    length = metadata >> METADATA_BITS
    msg_int = (point.x - metadata) // k
    if length == 0 and msg_int == 0:
        return b""
    if length == 0 and msg_int != 0:
        raise ValueError("Encoded point carries inconsistent length metadata")
    if length > METADATA_MASK:
        raise ValueError("Encoded length exceeds supported maximum")
    if msg_int.bit_length() > 8 * length:
        raise ValueError("Encoded point is malformed for the configured encoder")
    return msg_int.to_bytes(length, "big") if length else b""

def generate_key_pair(curve: EllipticCurve) -> int | ECPoint | None:
    private_key = secrets.randbelow(curve.n - 1) + 1
    public_key = scalar_mult(private_key, curve.g, curve)
    return private_key, public_key

def max_payload_length(curve: EllipticCurve, k: int = ENCODING_K) -> int:
    max_len = 0
    while True:
        candidate_len = max_len + 1
        if candidate_len > METADATA_MASK:
            break
        max_int = (1 << (8 * candidate_len)) - 1
        max_meta = (candidate_len << METADATA_BITS) | METADATA_MASK
        if max_int * k + max_meta >= curve.p:
            break
        max_len = candidate_len
    return max_len

def validate_public_key(pub: ECPoint, curve: EllipticCurve) -> None:
    if not curve.is_on_curve(pub):
        raise ValueError("Point not on curve")
    if scalar_mult(curve.n, pub, curve) != POINT_INFINITY:
        raise ValueError("Public key has invalid order (not in correct subgroup)")


def encrypt(message: bytes, public_key: ECPoint, curve: EllipticCurve, k: int = ENCODING_K) -> Ciphertext:
    validate_public_key(public_key, curve)
    chunk_len = max_payload_length(curve, k)
    if chunk_len == 0:
        raise ValueError("Encoding parameters are incompatible with the curve")

    chunks = [message[i : i + chunk_len] for i in range(0, len(message), chunk_len)] or [b""]
    ciphertext: Ciphertext = []
    for chunk in chunks:
        msg_point = encode_message(chunk, curve, k)
        ephemeral_key = secrets.randbelow(curve.n - 1) + 1
        c1 = scalar_mult(ephemeral_key, curve.g, curve)
        shared = scalar_mult(ephemeral_key, public_key, curve)
        c2 = point_add(msg_point, shared, curve)
        ciphertext.append((c1, c2))
    return ciphertext


def decrypt(ciphertext: Ciphertext, private_key: int, curve: EllipticCurve, k: int = ENCODING_K) -> bytes:
    plaintext_parts: list[bytes] = []
    for c1, c2 in ciphertext:
        if not (curve.is_on_curve(c1) and curve.is_on_curve(c2)):
            raise ValueError("Ciphertext points are not on the curve")
        shared = scalar_mult(private_key, c1, curve)
        if shared.is_infinity or shared.x is None:
            raise ValueError("Invalid shared secret computed")
        msg_point = point_add(c2, point_neg(shared, curve), curve)
        plaintext_parts.append(decode_message(msg_point, k))
    return b"".join(plaintext_parts)

def format_point(point: ECPoint) -> str:
    if point.is_infinity:
        return "Point(infinity)"
    assert point.x is not None and point.y is not None
    return f"Point(x={point.x:#x}, y={point.y:#x})"


def run_demo(message: str) -> None:
    curve = secp256k1()
    private_key, public_key = generate_key_pair(curve)

    plaintext = message.encode("utf-8")
    ciphertext = encrypt(plaintext, public_key, curve)
    recovered = decrypt(ciphertext, private_key, curve)
    print(ciphertext)
    print(recovered)
    print(plaintext)
    if recovered != plaintext:
        raise RuntimeError("Decryption failed: recovered plaintext does not match input")

    print("Curve: secp256k1")
    print(f"Private key: {private_key:#x}")
    print(f"Public key: {format_point(public_key)}")
    for idx, (c1, c2) in enumerate(ciphertext, start=1):
        print(f"Ciphertext block {idx} C1: {format_point(c1)}")
        print(f"Ciphertext block {idx} C2: {format_point(c2)}")
    print(f"Recovered plaintext: {recovered.decode('utf-8', errors='replace')}")

def main() -> None:
    parser = argparse.ArgumentParser(description="Elliptic-curve ElGamal encryption demo")
    parser.add_argument(
        "message",
        nargs="?",
        default="Привет, эллиптические",
        help="Message to encrypt and decrypt during the demo",
    )
    run_demo("2")


if __name__ == "__main__":
    main()

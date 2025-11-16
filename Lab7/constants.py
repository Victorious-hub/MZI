from dataclass import ECPoint

ENCODING_K = 1 << 16
METADATA_BITS = 8
METADATA_MASK = (1 << METADATA_BITS) - 1

POINT_INFINITY = ECPoint(None, None)

Ciphertext = list[tuple[ECPoint, ECPoint]]

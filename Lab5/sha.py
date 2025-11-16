from __future__ import annotations
from typing import Optional, Union


_SHA1InitialState = tuple[int, int, int, int, int]
BufferLike = Union[bytes, bytearray, memoryview]

def _to_bytes(data: BufferLike) -> bytes:
	"""Return the input as bytes, raising ``TypeError`` for unsupported types."""

	if isinstance(data, (bytes, bytearray)):
		return bytes(data)

	if isinstance(data, memoryview):
		if data.format not in ("B", "b"):
			raise TypeError("memoryview must be of a byte-oriented format")
		return data.tobytes()

	raise TypeError("data must be bytes-like")


class SHA1:
	_h0: int = 0x67452301
	_h1: int = 0xEFCDAB89
	_h2: int = 0x98BADCFE
	_h3: int = 0x10325476
	_h4: int = 0xC3D2E1F0
	_unprocessed: bytes = b""
	_message_byte_length: int = 0

	block_size: int = 64
	digest_size: int = 20

	def __init__(self, data: BufferLike | None = None):
		self._h0 = 0x67452301
		self._h1 = 0xEFCDAB89
		self._h2 = 0x98BADCFE
		self._h3 = 0x10325476
		self._h4 = 0xC3D2E1F0
		self._unprocessed = b""
		self._message_byte_length = 0

		if data is not None:
			self.update(data)

	def update(self, data: BufferLike) -> "SHA1":
		chunk = _to_bytes(data)
		self._message_byte_length += len(chunk)

		buffer = self._unprocessed + chunk
		for offset in range(0, len(buffer) - len(buffer) % self.block_size, self.block_size):
			block = buffer[offset : offset + self.block_size]
			self._process_block(block)

		self._unprocessed = buffer[len(buffer) - len(buffer) % self.block_size :]
		return self

	def digest(self) -> bytes:
		h0, h1, h2, h3, h4 = self._h0, self._h1, self._h2, self._h3, self._h4
		unprocessed = self._unprocessed
		total_length = self._message_byte_length

		# Append the bit '1' to the message
		final_message = unprocessed + b"\x80"

		padding_needed = (56 - len(final_message) % self.block_size) % self.block_size
		final_message += b"\x00" * padding_needed

		message_bit_length = total_length * 8
		final_message += message_bit_length.to_bytes(8, "big")

		for offset in range(0, len(final_message), self.block_size):
			block = final_message[offset : offset + self.block_size]
			h0, h1, h2, h3, h4 = self._process_block(block, (h0, h1, h2, h3, h4))

		return b"".join(word.to_bytes(4, "big") for word in (h0, h1, h2, h3, h4))

	def hexdigest(self) -> str:
		return self.digest().hex()

	def copy(self) -> "SHA1":
		clone = SHA1()
		clone._h0, clone._h1, clone._h2, clone._h3, clone._h4 = (
			self._h0,
			self._h1,
			self._h2,
			self._h3,
			self._h4,
		)
		clone._unprocessed = self._unprocessed
		clone._message_byte_length = self._message_byte_length
		return clone

	@classmethod
	def hash(cls, data: BufferLike) -> bytes:
		"""Return the SHA-1 digest for ``data`` as raw bytes."""
		return cls(data).digest()

	@classmethod
	def hexdigest_from(cls, data: BufferLike) -> str:
		"""Return the SHA-1 digest for ``data`` as a hex string."""
		return cls(data).hexdigest()

	def _process_block(
		self,
		block: bytes,
		state: Optional[_SHA1InitialState] = None,
	) -> _SHA1InitialState:
		if len(block) != self.block_size:
			raise ValueError("Block size must be exactly 64 bytes")

		if state is None:
			state = (self._h0, self._h1, self._h2, self._h3, self._h4)
			update_instance_state = True
		else:
			update_instance_state = False

		w = [0] * 80
		for i in range(16):
			w[i] = int.from_bytes(block[i * 4 : (i + 1) * 4], "big")

		for i in range(16, 80):
			w[i] = self._left_rotate(w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16], 1)

		a, b, c, d, e = state

		for i in range(80):
			if 0 <= i <= 19:
				f = (b & c) | ((~b & 0xFFFFFFFF) & d)
				k = 0x5A827999
			elif 20 <= i <= 39:
				f = b ^ c ^ d
				k = 0x6ED9EBA1
			elif 40 <= i <= 59:
				f = (b & c) | (b & d) | (c & d)
				k = 0x8F1BBCDC
			else:
				f = b ^ c ^ d
				k = 0xCA62C1D6

			temp = (self._left_rotate(a, 5) + f + e + k + w[i]) & 0xFFFFFFFF
			e = d
			d = c
			c = self._left_rotate(b, 30)
			b = a
			a = temp

		state_update = (
			(state[0] + a) & 0xFFFFFFFF,
			(state[1] + b) & 0xFFFFFFFF,
			(state[2] + c) & 0xFFFFFFFF,
			(state[3] + d) & 0xFFFFFFFF,
			(state[4] + e) & 0xFFFFFFFF,
		)
		if update_instance_state:
			self._h0, self._h1, self._h2, self._h3, self._h4 = state_update
		return state_update

	@staticmethod
	def _left_rotate(value: int, count: int) -> int:
		return ((value << count) | (value >> (32 - count))) & 0xFFFFFFFF


def sha1(data: BufferLike) -> bytes:
	return SHA1.hash(data)


def sha1_hex(data: BufferLike) -> str:
	return SHA1.hexdigest_from(data)


def hash_file(input_path: str, output_path: str, chunk_size: int = 8192) -> str:
	hasher = SHA1()
	with open(input_path, "rb") as src:
		while True:
			chunk = src.read(chunk_size)
			if not chunk:
				break
			hasher.update(chunk)

	hex_digest = hasher.hexdigest()
	with open(output_path, "w", encoding="utf-8") as dst:
		dst.write(hex_digest)
		dst.write("\n")

	return hex_digest

def sha1_text(text: str, encoding: str = "utf-8") -> str:
    data = text.encode(encoding)
    return SHA1(data).hexdigest()

if __name__ == "__main__":
    s = ""
    print("Текст:", repr(s))
    print("SHA-1:", sha1_text(s))

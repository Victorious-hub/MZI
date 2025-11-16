import argparse
import struct
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image


BLOCK_SIZE = 8
QUANT_MATRIX = np.array(
	[
		16,11,10,16,24,40,51,61,12,12,14,19,26,58,60,55,14,13,16,
		24,40,57,69,56,14,17,22,29,51,87,80,62,18,22,37,56,68,109,
		103,77,24,35,55,64,81,104,113,92,49,64,78,87,103,121,120,101,72,92,95,98,112,100,103,99,
	],
	dtype=np.int32,
).reshape((BLOCK_SIZE, BLOCK_SIZE))

STD_CHROMA_Q = np.array(
	[
		17,18,24,47,99,99,99,99,18,21,26,66,99,99,99,99,24,26,56,
		99,99,99,99,99,47,66,99,99,99,99,99,99,99,99,99,99,99,99,
		99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,
		99,99,99,99,99,99,99,
	],
	dtype=np.int32,
).reshape((BLOCK_SIZE, BLOCK_SIZE))

def build_dct_matrix(n: int = BLOCK_SIZE) -> np.ndarray:
	mat = np.zeros((n, n), dtype=np.float64)
	factor = np.pi / (2.0 * n)
	scale = np.sqrt(1.0 / n)
	mat[0, :] = scale
	for i in range(1, n):
		mat[i, :] = np.sqrt(2.0 / n) * np.cos((2 * np.arange(n) + 1) * i * factor)
	return mat


DCT_MATRIX = build_dct_matrix()

def dct2(block: np.ndarray) -> np.ndarray:
	return DCT_MATRIX @ block @ DCT_MATRIX.T

def idct2(block: np.ndarray) -> np.ndarray:
	return DCT_MATRIX.T @ block @ DCT_MATRIX

def scale_quant_table(table: np.ndarray, quality: int) -> np.ndarray:
	if not (1 <= quality <= 100):
		raise ValueError("Quality must be within [1, 100]")
	if quality < 50:
		scale = 5000 / quality
	else:
		scale = 200 - quality * 2
	scaled = np.floor((table * scale + 50) / 100)
	scaled[scaled < 1] = 1
	scaled[scaled > 255] = 255
	return scaled.astype(np.int32)

def pad_channel(channel: np.ndarray) -> Tuple[np.ndarray, int, int]:
	height, width = channel.shape
	pad_h = (BLOCK_SIZE - height % BLOCK_SIZE) % BLOCK_SIZE
	pad_w = (BLOCK_SIZE - width % BLOCK_SIZE) % BLOCK_SIZE
	padded = np.pad(channel, ((0, pad_h), (0, pad_w)), mode="edge")
	return padded.astype(np.float64), pad_h, pad_w

def iter_block_coords(height: int, width: int) -> Iterable[Tuple[int, int]]:
	for y in range(0, height, BLOCK_SIZE):
		for x in range(0, width, BLOCK_SIZE):
			yield y, x


def message_to_bits(data: bytes) -> List[int]:
	bits: List[int] = []
	for byte in data:
		for shift in range(7, -1, -1):
			bits.append((byte >> shift) & 1)
	return bits


def bits_to_bytes(bits: Sequence[int]) -> bytes:
	if len(bits) % 8 != 0:
		raise ValueError("Bit sequence length must be a multiple of 8")
	bytelist = []
	for idx in range(0, len(bits), 8):
		byte_val = 0
		for bit in bits[idx : idx + 8]:
			byte_val = (byte_val << 1) | bit
		bytelist.append(byte_val)
	return bytes(bytelist)

def embed_bits_in_channel(
	channel: np.ndarray,
	bits: Sequence[int],
	coefficient: Tuple[int, int],
	quant_table: np.ndarray,
) -> np.ndarray:
	padded, pad_h, pad_w = pad_channel(channel)
	height, width = padded.shape
	total_blocks = (height // BLOCK_SIZE) * (width // BLOCK_SIZE)
	if len(bits) > total_blocks:
		raise ValueError(
			f"Message too large: need {len(bits)} blocks, have {total_blocks}"
		)

	result = padded.copy()
	bit_idx = 0
	cy, cx = coefficient

	for y, x in iter_block_coords(height, width):
		if bit_idx >= len(bits):
			break
		block = result[y : y + BLOCK_SIZE, x : x + BLOCK_SIZE]
		dct_block = dct2(block)
		q_block = np.round(dct_block / quant_table)
		coeff_int = int(q_block[cy, cx])
		desired_bit = bits[bit_idx]
		if (coeff_int & 1) != desired_bit:
			coeff_int += 1 if coeff_int >= 0 else -1
			if coeff_int == 0:
				coeff_int = 1 if desired_bit else -1
		q_block[cy, cx] = coeff_int
		modified = q_block * quant_table
		result[y : y + BLOCK_SIZE, x : x + BLOCK_SIZE] = idct2(modified)
		bit_idx += 1

	if bit_idx < len(bits):
		raise RuntimeError("Failed to embed all message bits")

	clipped = np.clip(np.round(result), 0, 255)
	if pad_h or pad_w:
		clipped = clipped[: channel.shape[0], : channel.shape[1]]
	return clipped.astype(np.uint8)


def extract_bits_from_channel(
	channel: np.ndarray,
	bit_count: int,
	coefficient: Tuple[int, int],
	quant_table: np.ndarray,
) -> List[int]:
	padded, _, _ = pad_channel(channel)
	height, width = padded.shape
	cy, cx = coefficient
	bits: List[int] = []

	for y, x in iter_block_coords(height, width):
		if len(bits) >= bit_count:
			break
		block = padded[y : y + BLOCK_SIZE, x : x + BLOCK_SIZE]
		q_block = np.round(dct2(block) / quant_table)
		bits.append(int(q_block[cy, cx]) & 1)

	if len(bits) < bit_count:
		raise ValueError("Not enough data in image to extract message")
	return bits


def load_ycbcr(image_path: Path) -> Tuple[np.ndarray, Image.Image, Image.Image]:
	with Image.open(image_path) as img:
		if img.format != "JPEG":
			raise ValueError(f"Unsupported image format: {img.format}. Please use a JPEG image.")
		ycbcr = img.convert("YCbCr")
		y, cb, cr = ycbcr.split()
		return np.array(y, dtype=np.uint8), cb, cr


def save_ycbcr(
	y_channel: np.ndarray,
	cb: Image.Image,
	cr: Image.Image,
	output_path: Path,
	quality: int,
	luma_q: np.ndarray,
	chroma_q: np.ndarray,
) -> None:
	y_img = Image.fromarray(y_channel, "L")
	merged = Image.merge("YCbCr", (y_img, cb, cr)).convert("RGB")
	qtables = [luma_q.flatten().tolist(), chroma_q.flatten().tolist()]
	merged.save(
		output_path,
		format="JPEG",
		quality=quality,
		subsampling=0,
		qtables=qtables,
	)


def prepare_message_bits(message: bytes) -> List[int]:
	header = struct.pack(">I", len(message))
	return message_to_bits(header + message)

def read_hidden_message(
	channel: np.ndarray,
	coefficient: Tuple[int, int],
	quant_table: np.ndarray,
) -> bytes:
	header_bits = extract_bits_from_channel(channel, 32, coefficient, quant_table)
	header_bytes = bits_to_bytes(header_bits)
	(length,) = struct.unpack(">I", header_bytes)
	if length == 0:
		return b""
	data_bits = extract_bits_from_channel(
		channel,
		32 + length * 8,
		coefficient,
		quant_table,
	)[32:]
	return bits_to_bytes(data_bits)


def hide_message(
	input_path: Path,
	output_path: Path,
	message: bytes,
	coefficient: Tuple[int, int],
	quality: int,
) -> None:
	y_channel, cb, cr = load_ycbcr(input_path)
	bits = prepare_message_bits(message)
	luma_q_int = scale_quant_table(QUANT_MATRIX, quality)
	chroma_q_int = scale_quant_table(STD_CHROMA_Q, quality)
	luma_q = luma_q_int.astype(np.float64)
	modified_y = embed_bits_in_channel(y_channel, bits, coefficient, luma_q)
	save_ycbcr(modified_y, cb, cr, output_path, quality, luma_q_int, chroma_q_int)


def extract_message(
	input_path: Path,
	coefficient: Tuple[int, int],
	quality: int,
) -> bytes:
	y_channel, _, _ = load_ycbcr(input_path)
	luma_q = scale_quant_table(QUANT_MATRIX, quality).astype(np.float64)
	return read_hidden_message(y_channel, coefficient, luma_q)


def parse_coefficient(text: str) -> Tuple[int, int]:
	parts = text.split(",")
	if len(parts) != 2:
		raise argparse.ArgumentTypeError("Coefficient must be 'row,col'")
	try:
		row = int(parts[0])
		col = int(parts[1])
	except ValueError as exc:  # pragma: no cover - user input guard
		raise argparse.ArgumentTypeError("Coefficient indices must be integers") from exc
	if not (0 <= row < BLOCK_SIZE and 0 <= col < BLOCK_SIZE):
		raise argparse.ArgumentTypeError("Coefficient indices must be within [0,7]")
	if row == 0 and col == 0:
		raise argparse.ArgumentTypeError("DC coefficient (0,0) is not allowed")
	return row, col


def read_message_source(args: argparse.Namespace) -> bytes:
	if getattr(args, "message", None) and getattr(args, "message_file", None):
		raise ValueError("Specify either --message or --message-file, not both")
	if getattr(args, "message", None):
		return args.message.encode("utf-8")
	if getattr(args, "message_file", None):
		return Path(args.message_file).read_bytes()
	raise ValueError("No message source provided")


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		description="Hide or extract a UTF-8 message inside a JPEG via frequency-domain steganography",
	)
	subparsers = parser.add_subparsers(dest="command", required=True)

	hide_parser = subparsers.add_parser("hide", help="Embed a message into an image")
	hide_parser.add_argument("input_image", type=Path)
	hide_parser.add_argument("output_image", type=Path)
	hide_parser.add_argument("--message", help="Literal UTF-8 string to hide")
	hide_parser.add_argument("--message-file", help="Path to a file whose contents will be hidden")
	hide_parser.add_argument(
		"--quality",
		type=int,
		default=95,
		help="JPEG quality for the stego image (default: 95)",
	)
	hide_parser.add_argument(
		"--coefficient",
		type=parse_coefficient,
		default=(3, 4),
		help="DCT coefficient indices as 'row,col' (default: 3,4)",
	)

	extract_parser = subparsers.add_parser("extract", help="Extract a hidden message")
	extract_parser.add_argument("input_image", type=Path)
	extract_parser.add_argument(
		"--coefficient",
		type=parse_coefficient,
		default=(3, 4),
		help="DCT coefficient indices used during embedding",
	)
	extract_parser.add_argument(
		"--quality",
		type=int,
		default=95,
		help="JPEG quality that was used during embedding (default: 95)",
	)
	extract_parser.add_argument(
		"--output",
		type=Path,
		help="Optional output file for the extracted message",
	)

	return parser


def main() -> None:
	parser = build_parser()
	args = parser.parse_args()

	if args.command == "hide":
		message = read_message_source(args)
		hide_message(
			input_path=args.input_image,
			output_path=args.output_image,
			message=message,
			coefficient=args.coefficient,
			quality=args.quality,
		)
	else:
		data = extract_message(
			input_path=args.input_image,
			coefficient=args.coefficient,
			quality=args.quality,
		)
		if args.output:
			args.output.write_bytes(data)
		else:
			print(data.decode("utf-8", errors="replace"))


if __name__ == "__main__":
	main()


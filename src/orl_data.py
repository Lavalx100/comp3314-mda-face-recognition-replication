"""Read the original-resolution AT&T/ORL PGM face archive without extracting it."""

from __future__ import annotations

import re
from pathlib import Path, PurePosixPath
from zipfile import BadZipFile, ZipFile

import numpy as np


ORL_ARCHIVE_URL = "https://www.cl.cam.ac.uk/research/dtg/attarchive/pub/data/att_faces.zip"
ORL_CITATION = "Olivetti Research Laboratory, The ORL Database of Faces (1992–1994)."


def _pgm_tokens(data: bytes):
    """Yield header/text tokens and leave the cursor at each token boundary."""
    cursor = 0
    length = len(data)

    def next_token() -> bytes:
        nonlocal cursor
        while cursor < length:
            byte = data[cursor]
            if byte in b" \t\r\n\v\f":
                cursor += 1
            elif byte == ord("#"):
                while cursor < length and data[cursor] not in b"\r\n":
                    cursor += 1
            else:
                break
        start = cursor
        while cursor < length and data[cursor] not in b" \t\r\n\v\f#":
            cursor += 1
        if start == cursor:
            raise ValueError("Unexpected end of PGM header or pixel data.")
        return data[start:cursor]

    while True:
        try:
            yield next_token()
        except ValueError:
            return


def read_pgm(data: bytes) -> np.ndarray:
    """Decode a P2/P5 PGM image to uint8 while respecting comments and maxval."""
    cursor = 0
    length = len(data)

    def token() -> bytes:
        nonlocal cursor
        while cursor < length:
            byte = data[cursor]
            if byte in b" \t\r\n\v\f":
                cursor += 1
            elif byte == ord("#"):
                while cursor < length and data[cursor] not in b"\r\n":
                    cursor += 1
            else:
                break
        start = cursor
        while cursor < length and data[cursor] not in b" \t\r\n\v\f#":
            cursor += 1
        if start == cursor:
            raise ValueError("Unexpected end of PGM header.")
        return data[start:cursor]

    magic = token()
    if magic not in (b"P2", b"P5"):
        raise ValueError(f"Unsupported PGM magic value: {magic!r}")
    width, height, max_value = int(token()), int(token()), int(token())
    if width < 1 or height < 1 or not 1 <= max_value <= 65535:
        raise ValueError(f"Invalid PGM dimensions or max value: {width}x{height}, {max_value}")

    if magic == b"P2":
        values = [int(value) for value in _pgm_tokens(data[cursor:])]
        if len(values) != width * height:
            raise ValueError(f"PGM has {len(values)} values; expected {width * height}.")
        pixels = np.asarray(values, dtype=np.uint16 if max_value > 255 else np.uint8)
    else:
        # P5 uses one whitespace byte after maxval; CRLF is one logical newline.
        if cursor >= length or data[cursor] not in b" \t\r\n\v\f":
            raise ValueError("Missing PGM header/pixel separator.")
        if data[cursor:cursor + 2] == b"\r\n":
            cursor += 2
        else:
            cursor += 1
        count = width * height
        if max_value < 256:
            dtype = np.uint8
            bytes_per_pixel = 1
        else:
            dtype = np.dtype(">u2")
            bytes_per_pixel = 2
        expected_bytes = count * bytes_per_pixel
        raster = data[cursor:cursor + expected_bytes]
        if len(raster) != expected_bytes:
            raise ValueError(f"PGM raster has {len(raster)} bytes; expected {expected_bytes}.")
        pixels = np.frombuffer(raster, dtype=dtype, count=count)

    pixels = np.asarray(pixels, dtype=np.float64)
    if np.any(pixels < 0) or np.any(pixels > max_value):
        raise ValueError("PGM pixel value lies outside its declared range.")
    pixels = np.rint(pixels * (255.0 / max_value)).astype(np.uint8)
    return pixels.reshape(height, width)


def _archive_members(archive: ZipFile) -> dict[tuple[int, int], str]:
    members: dict[tuple[int, int], str] = {}
    for name in archive.namelist():
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"Unsafe path in ORL archive: {name!r}")
        if path.suffix.lower() != ".pgm" or len(path.parts) < 2:
            continue
        subject_match = re.fullmatch(r"s(\d+)", path.parts[-2], flags=re.IGNORECASE)
        image_match = re.fullmatch(r"(\d+)\.pgm", path.name, flags=re.IGNORECASE)
        if not subject_match or not image_match:
            continue
        key = int(subject_match.group(1)), int(image_match.group(1))
        if key in members:
            raise ValueError(f"Duplicate ORL image in archive: subject {key[0]}, image {key[1]}")
        members[key] = name

    expected = {(subject, image) for subject in range(1, 41) for image in range(1, 11)}
    missing = sorted(expected - members.keys())
    unexpected = sorted(members.keys() - expected)
    if missing or unexpected:
        raise ValueError(
            f"Archive must contain s1..s40/1..10.pgm; missing={missing[:5]}, "
            f"unexpected={unexpected[:5]}"
        )
    return members


def load_original_orl(archive_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load original 112-row × 92-column ORL images and integer identity labels."""
    archive_path = Path(archive_path)
    if not archive_path.is_file():
        raise FileNotFoundError(
            f"Original ORL archive not found at {archive_path}. Download it from "
            f"{ORL_ARCHIVE_URL} and keep it in the local cache; do not commit the faces."
        )
    try:
        with ZipFile(archive_path) as archive:
            members = _archive_members(archive)
            images: list[np.ndarray] = []
            labels: list[int] = []
            for subject in range(1, 41):
                for image_number in range(1, 11):
                    image = read_pgm(archive.read(members[(subject, image_number)]))
                    if image.shape != (112, 92):
                        raise ValueError(
                            f"Unexpected shape for s{subject}/{image_number}.pgm: {image.shape}; "
                            "expected (112, 92)."
                        )
                    images.append(image.astype(np.float64) / 255.0)
                    labels.append(subject - 1)
    except BadZipFile as exc:
        raise ValueError(f"Not a valid ORL ZIP archive: {archive_path}") from exc

    return np.stack(images), np.asarray(labels, dtype=np.int64)

"""Pinned public-domain text for offline model input preparation."""

import hashlib
from importlib.resources import files
from pathlib import Path

SHA256 = "ee916c3a0ce79fc528621480fd5123f9b55258f0e5938a6be6a04a0defeac06d"
SOURCE = "https://www.gutenberg.org/ebooks/11"


def bundled_text(destination):
    data = files("faultline").joinpath("data/alice.txt").read_bytes()
    if hashlib.sha256(data).hexdigest() != SHA256:
        raise ValueError("Bundled public-domain text checksum mismatch; reinstall Faultline")
    Path(destination).write_bytes(data)
    return {
        "url": SOURCE,
        "sha256": SHA256,
        "bytes": len(data),
        "transport": "bundled",
        "work": "Alice's Adventures in Wonderland (Lewis Carroll, 1865)",
    }

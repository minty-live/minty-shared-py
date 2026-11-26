"""Photos: compressed for storage, HEIC/HEIF turned into JPEG.

``for_storage`` (the user, 2026-11-26) is what an upload keeps in B2: a photo (JPEG, HEIC/HEIF,
WebP, BMP, TIFF, and a PNG when that is smaller) as an upright JPEG, at most 2048 px on its
longest side, quality 85, no EXIF; documents exactly as uploaded. Every type it stores is one
Xero takes, so the stored file is what is shown and published.

An iPhone photo can arrive as HEIC (from the Files app, a Mac, a PC copy; an Android phone set to
"high-efficiency" photos saves HEIF). Since 2026-11-26 it is stored as JPEG. HEIC uploaded
between 2026-11-24 and then is still in B2 as it was, so a JPEG is made on the fly for those:

* to SHOW it, when the person's browser can't draw HEIC (Safari 17+ can; Chrome, Edge, Firefox
  and Android can't) - the services' preview routes answer ``?as=jpeg`` with ``to_jpeg``;
* to PUBLISH it, because Xero takes no HEIC: ``for_xero`` turns a HEIC into ``<stem>.jpg`` and
  passes every other file through unchanged;
* to READ it with AI: ``minty_shared.model_input`` decodes it like any other photo.

``check_decodes`` is the upload's check, so a HEIC that can't be opened is refused when it is
attached, never found out at publish.

Decoding is ``pi-heif`` (decode-only, no encoder). Pillow and it are imported on first use, so
``sniff_heif`` and ``is_heif`` cost nothing to import. Extra ``images`` (``ai`` brings it too).
"""

from __future__ import annotations

import io
import threading
from pathlib import PurePosixPath

HEIC = "image/heic"
HEIF = "image/heif"
HEIF_MIMES = frozenset({HEIC, HEIF})
JPEG = "image/jpeg"

#: The longest side of a JPEG made here: keeps a 48 MP photo well under Xero's 10 MB.
DEFAULT_MAX_SIDE = 4096
JPEG_QUALITY = 90
#: Pillow's decompression-bomb guard, made explicit (as in ``model_input``).
MAX_DECODE_PIXELS = 80_000_000
#: A stored photo (``for_storage``; the user, 2026-11-26): the AI copy's size, ~0.3-1 MB.
STORE_MAX_SIDE = 2048
STORE_QUALITY = 85

_JPEG_MIMES = frozenset({JPEG, "image/jpg", "image/pjpeg"})
#: Stored as JPEG whatever the size: Xero refuses WebP (and HEIC); BMP/TIFF are uncompressed.
_ALWAYS_JPEG = frozenset({"image/webp", "image/bmp", "image/x-ms-bmp", "image/tiff"})

# ISO base media file "ftyp" major brands (ISO/IEC 23008-12). AVIF ("avif") is not HEIC.
_HEIC_BRANDS = frozenset({b"heic", b"heix", b"hevc", b"hevx", b"heim", b"heis", b"hevm", b"hevs"})
_HEIF_BRANDS = frozenset({b"mif1", b"msf1"})

_registered = False
_register_lock = threading.Lock()


class ImageError(Exception):
    """The photo could not be decoded. The cause is chained; the caller answers with its own
    sentence."""


def sniff_heif(head: bytes) -> str | None:
    """``image/heic``, ``image/heif`` or None, judged by the first 12+ bytes."""
    if len(head) < 12 or head[4:8] != b"ftyp":
        return None
    brand = bytes(head[8:12])
    if brand in _HEIC_BRANDS:
        return HEIC
    if brand in _HEIF_BRANDS:
        return HEIF
    return None


def is_heif(mime: str | None) -> bool:
    return (mime or "").strip().lower() in HEIF_MIMES


def register() -> None:
    """Teach Pillow to open HEIC/HEIF. Idempotent; ``model_input`` calls it too."""
    global _registered
    if _registered:
        return
    with _register_lock:
        if not _registered:
            from pi_heif import register_heif_opener

            register_heif_opener()
            _registered = True


def to_jpeg(data: bytes, *, max_side: int = DEFAULT_MAX_SIDE, quality: int = JPEG_QUALITY) -> bytes:
    """``data`` (any image Pillow or pi-heif opens) as an upright JPEG, at most ``max_side`` on its
    longest side, transparency flattened onto white, no EXIF kept. Raises ``ImageError`` when it
    won't decode."""
    return _encode(data, max_side, quality)[0]


def _encode(data: bytes, max_side: int, quality: int) -> tuple[bytes, tuple[int, int]]:
    """``(jpeg, original size)``. A JPEG at least twice the final size is decoded straight at 1/2,
    1/4 or 1/8 scale (Pillow's ``draft``, as in ``model_input``), so a 48 MP photo never becomes a
    144 MB bitmap on a 512 MB server."""
    from PIL import Image, ImageOps

    register()
    Image.MAX_IMAGE_PIXELS = MAX_DECODE_PIXELS
    try:
        with Image.open(io.BytesIO(data)) as opened:
            original = opened.size
            sw, sh = original
            scale = min(1.0, max_side / max(sw, sh))
            opened.draft("RGB", (max(1, round(sw * scale)), max(1, round(sh * scale))))
            img = ImageOps.exif_transpose(opened)
            w, h = img.size
            scale = min(1.0, max_side / max(w, h))
            if scale < 1.0:
                size = (max(1, round(w * scale)), max(1, round(h * scale)))
                img = img.resize(size, Image.Resampling.LANCZOS)
            if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
                rgba = img.convert("RGBA")
                flat = Image.new("RGB", rgba.size, (255, 255, 255))
                flat.paste(rgba, mask=rgba.getchannel("A"))
            else:
                flat = img.convert("RGB")
            out = io.BytesIO()
            flat.save(out, format="JPEG", quality=quality)
            return out.getvalue(), original
    except Exception as exc:  # noqa: BLE001 - any decoder failure is the same refusal
        raise ImageError(f"The image could not be decoded: {exc}") from exc


def check_decodes(data: bytes) -> None:
    """Raise ``ImageError`` unless ``data`` decodes as an image (the upload's check)."""
    from PIL import Image

    register()
    Image.MAX_IMAGE_PIXELS = MAX_DECODE_PIXELS
    try:
        with Image.open(io.BytesIO(data)) as img:
            img.load()
    except Exception as exc:  # noqa: BLE001 - any decoder failure is the same refusal
        raise ImageError(f"The image could not be decoded: {exc}") from exc


def jpeg_name(name: str) -> str:
    """``IMG_1234.HEIC`` -> ``IMG_1234.jpg``; a name with no stem becomes ``photo.jpg``."""
    stem = PurePosixPath(name.replace("\\", "/")).stem if name else ""
    return f"{stem or 'photo'}.jpg"


def for_xero(data: bytes, mime: str, name: str) -> tuple[bytes, str, str]:
    """``(bytes, mime, name)`` to send to Xero: a HEIC/HEIF (by its type or its bytes) as a JPEG
    named ``<stem>.jpg``, because Xero takes no HEIC; any other file exactly as it is."""
    if is_heif(mime) or sniff_heif(data[:16]):
        return to_jpeg(data), JPEG, jpeg_name(name)
    return data, mime, name


def for_storage(data: bytes, mime: str, name: str) -> tuple[bytes, str, str]:
    """``(bytes, mime, name)`` to keep in B2 for an upload (the user, 2026-11-26): photos are
    stored as a JPEG at most ``STORE_MAX_SIDE`` on the longest side, in a type Xero takes, so the
    stored file is what is shown and what is published.

    * HEIC/HEIF (by type or bytes), WebP, BMP, TIFF: always the JPEG (Xero takes no HEIC or WebP;
      BMP and TIFF are many times the size);
    * JPEG: the JPEG, except that one already within the cap is kept when re-saving would not
      make it smaller (no second loss for nothing);
    * PNG: the JPEG only when it is smaller - a sharp screenshot stays PNG;
    * anything else (PDF, Office files, CSV, text, HTML, GIF, SVG): exactly as uploaded.

    A photo that won't decode raises ``ImageError``: it is refused, never stored broken."""
    kind = (mime or "").strip().lower()
    if is_heif(kind) or sniff_heif(data[:16]) or kind in _ALWAYS_JPEG:
        return _encode(data, STORE_MAX_SIDE, STORE_QUALITY)[0], JPEG, jpeg_name(name)
    if kind in _JPEG_MIMES or kind == "image/png":
        jpeg, (w, h) = _encode(data, STORE_MAX_SIDE, STORE_QUALITY)
        within_cap = max(w, h) <= STORE_MAX_SIDE
        if len(jpeg) >= len(data) and (within_cap or kind == "image/png"):
            return data, mime, name
        return jpeg, JPEG, jpeg_name(name)
    return data, mime, name

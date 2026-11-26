"""The copy of a document a model reads - never the stored file (owner, 2026-11-04).

The reader works on the uploaded bytes, never on B2's stored copy. Before a receipt or
invoice is sent to the model, ``prepare_for_model`` makes the small copy it is billed for:

* an image is capped at ``max_side`` on its longest side, flattened onto white and saved ONCE
  as JPEG. A JPEG at least twice the final size is decoded straight at 1/2, 1/4 or 1/8 scale
  (Pillow's ``draft``), so a 48 MP photo is read as 12 MP, never as a 144 MB bitmap; a smaller
  one is decoded once at full size (a 12 MP photo is ~36 MB, freed at once). The model is billed by the image's pixels, not its bytes,
  so there is no byte target and no second pass;
* a PDF is sent whole, every page, as it is (the user, 2026-11-24: an invoice's total, tax and
  bank details are often on its last page). It is billed per page; the callers hold it to 10 MB
  (`ai_policy.max_file_mb` and each route's upload check). It is never opened here.

It replaces Flask's ``file_downsize.py``, whose seven-pass, 1 MB-target loop was the CPU and RAM
cost of reading a receipt. Never raises: on any failure it logs and returns the input unchanged -
a slightly larger request beats no suggestion. Extra ``ai``.
"""

from __future__ import annotations

import io
import logging

from minty_shared import images

logger = logging.getLogger("minty-shared.model_input")

DEFAULT_MAX_SIDE = 2048
JPEG_QUALITY = 85
#: Pillow's decompression-bomb guard, made explicit: an image above this raises on decode.
MAX_DECODE_PIXELS = 80_000_000

#: heic/heif: an iPhone photo kept as HEIC (the user, 2026-11-24), decoded through
#: ``minty_shared.images`` (pi-heif) and sent as JPEG like any other photo.
_IMAGES = {"image/jpeg", "image/jpg", "image/pjpeg", "image/png", *images.HEIF_MIMES}


def prepare_for_model(
    data: bytes, mime: str, *, max_side: int = DEFAULT_MAX_SIDE
) -> tuple[bytes, str]:
    """``(bytes, mime)`` to send to the model for ``data`` of type ``mime``."""
    if not data:
        return data, mime
    try:
        if mime in _IMAGES:
            return _image(data, max_side), "image/jpeg"
    except Exception as exc:  # noqa: BLE001 - best effort by design; logged, input returned
        logger.warning(
            "prepare_for_model: sending the original %s (%s bytes): %s", mime, len(data), exc
        )
    return data, mime


def _image(data: bytes, max_side: int) -> bytes:
    from PIL import Image

    images.register()
    Image.MAX_IMAGE_PIXELS = MAX_DECODE_PIXELS
    with Image.open(io.BytesIO(data)) as img:
        w, h = img.size
        scale = min(1.0, max_side / max(w, h))
        final = (max(1, round(w * scale)), max(1, round(h * scale)))
        # draft() makes a JPEG decode at the largest 1/2, 1/4 or 1/8 scale still at least `final`
        # (a no-op for other formats). Called here with the aspect-correct size because Pillow 11's
        # thumbnail() asks it for a square, which a landscape photo never meets.
        img.draft("RGB", final)
        if img.size != final:
            img = img.resize(final, Image.Resampling.LANCZOS)
        if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
            rgba = img.convert("RGBA")
            flat = Image.new("RGB", rgba.size, (255, 255, 255))
            flat.paste(rgba, mask=rgba.getchannel("A"))
        else:
            flat = img.convert("RGB")
        out = io.BytesIO()
        flat.save(out, format="JPEG", quality=JPEG_QUALITY)
        return out.getvalue()

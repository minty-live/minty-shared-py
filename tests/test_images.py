"""minty_shared.images: HEIC/HEIF photos stored as uploaded, shown and published as JPEG only when
the browser or Xero can't take them (the user, 2026-11-24)."""

import io
from pathlib import Path

import pytest

PIL = pytest.importorskip("PIL")
pytest.importorskip("pi_heif")

from PIL import Image  # noqa: E402

from minty_shared import images  # noqa: E402

#: 80x40 as stored with EXIF orientation 6, left half red and right half blue: upright it is
#: 40x80, red on top. Made once with pillow-heif (tests/fixtures/tiny.heic, 680 bytes).
HEIC = (Path(__file__).parent / "fixtures" / "tiny.heic").read_bytes()


def _decoded(data: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(data))
    img.load()
    return img


def _png(width, height, mode="RGB", colour=(10, 120, 200)) -> bytes:
    buf = io.BytesIO()
    Image.new(mode, (width, height), colour).save(buf, format="PNG")
    return buf.getvalue()


# --------------------------------------------------------------------------- sniffing


@pytest.mark.parametrize("brand", [b"heic", b"heix", b"hevc", b"hevx", b"heim", b"heis"])
def test_heic_brands_are_heic(brand):
    assert images.sniff_heif(b"\x00\x00\x00\x18ftyp" + brand + b"\x00\x00\x00\x00") == "image/heic"


@pytest.mark.parametrize("brand", [b"mif1", b"msf1"])
def test_the_general_heif_brands_are_heif(brand):
    assert images.sniff_heif(b"\x00\x00\x00\x18ftyp" + brand + b"\x00\x00\x00\x00") == "image/heif"


def test_the_fixture_sniffs_as_heic():
    assert images.sniff_heif(HEIC[:16]) == "image/heic"


@pytest.mark.parametrize(
    "head",
    [
        b"",
        b"\x00\x00\x00\x18ftyp",  # too short to carry a brand
        b"\x00\x00\x00\x18ftypisom\x00\x00\x00\x00",  # an MP4 video
        b"\x00\x00\x00\x1cftypavif\x00\x00\x00\x00",  # AVIF: not HEIC
        b"\xff\xd8\xff\xe0 a jpeg",
        b"%PDF-1.7",
    ],
)
def test_other_files_are_not_heif(head):
    assert images.sniff_heif(head) is None


@pytest.mark.parametrize(
    ("mime", "expected"),
    [("image/heic", True), ("IMAGE/HEIF", True), ("image/jpeg", False), ("", False), (None, False)],
)
def test_is_heif(mime, expected):
    assert images.is_heif(mime) is expected


# --------------------------------------------------------------------------- to_jpeg


def test_a_heic_becomes_an_upright_jpeg():
    out = images.to_jpeg(HEIC)
    img = _decoded(out)
    assert (img.format, img.mode, img.size) == ("JPEG", "RGB", (40, 80))
    r, g, b = img.getpixel((20, 5))  # top: red
    assert r > 180 and b < 80
    r, g, b = img.getpixel((20, 75))  # bottom: blue
    assert b > 180 and r < 80


def test_a_large_image_is_capped_on_its_longest_side():
    img = _decoded(images.to_jpeg(_png(5000, 100), max_side=4096))
    assert img.size == (4096, 82)


def test_a_small_image_is_not_upscaled():
    assert _decoded(images.to_jpeg(_png(300, 200))).size == (300, 200)


def test_transparency_is_flattened_onto_white():
    img = _decoded(images.to_jpeg(_png(20, 20, mode="RGBA", colour=(0, 0, 0, 0))))
    assert min(img.getpixel((10, 10))) > 240


def test_bytes_that_do_not_decode_raise_image_error():
    with pytest.raises(images.ImageError):
        images.to_jpeg(b"\x00\x00\x00\x18ftypheic not really a photo")


def test_check_decodes_passes_a_real_heic_and_refuses_junk():
    images.check_decodes(HEIC)
    with pytest.raises(images.ImageError):
        images.check_decodes(b"\x00\x00\x00\x18ftypheic not really a photo")


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("IMG_1234.HEIC", "IMG_1234.jpg"),
        ("photo.heif", "photo.jpg"),
        ("no-extension", "no-extension.jpg"),
        ("", "photo.jpg"),
    ],
)
def test_jpeg_name(name, expected):
    assert images.jpeg_name(name) == expected


# --------------------------------------------------------------------------- for_xero


def test_a_heic_goes_to_xero_as_jpeg_named_jpg():
    data, mime, name = images.for_xero(HEIC, "image/heic", "IMG_1234.HEIC")
    assert (mime, name) == ("image/jpeg", "IMG_1234.jpg")
    assert _decoded(data).format == "JPEG"


def test_a_heic_is_known_by_its_bytes_whatever_it_was_called():
    data, mime, name = images.for_xero(HEIC, "application/octet-stream", "slip.bin")
    assert (mime, name) == ("image/jpeg", "slip.jpg")
    assert _decoded(data).format == "JPEG"


@pytest.mark.parametrize(
    ("data", "mime", "name"),
    [
        (b"\xff\xd8\xff\xe0 jpeg bytes", "image/jpeg", "a.jpg"),
        (b"\x89PNG\r\n\x1a\n png bytes", "image/png", "a.png"),
        (b"%PDF-1.7 pdf bytes", "application/pdf", "a.pdf"),
        (b"<html></html>", "text/html", "a.html"),
    ],
)
def test_types_xero_takes_go_unchanged(data, mime, name):
    assert images.for_xero(data, mime, name) == (data, mime, name)


# --------------------------------------------------------------------------- for_storage


def _photo(width, height, fmt="JPEG", **save) -> bytes:
    """A photo-like image (noise compresses like a camera picture, not like a flat colour)."""
    import random

    rnd = random.Random(width * 7 + height)
    img = Image.frombytes("RGB", (width, height), rnd.randbytes(width * height * 3))
    buf = io.BytesIO()
    img.save(buf, format=fmt, **save)
    return buf.getvalue()


def test_a_large_jpeg_is_stored_as_a_2048_jpeg():
    original = _photo(4032, 3024, quality=95)
    data, mime, name = images.for_storage(original, "image/jpeg", "IMG_1.JPG")
    img = _decoded(data)
    assert (img.format, img.size, mime, name) == ("JPEG", (2048, 1536), "image/jpeg", "IMG_1.jpg")
    assert len(data) < len(original)


def test_a_small_jpeg_that_would_not_shrink_is_kept_as_it_is():
    original = _photo(800, 600, quality=60)
    assert images.for_storage(original, "image/jpeg", "a.jpeg") == (
        original,
        "image/jpeg",
        "a.jpeg",
    )


def test_a_heic_is_stored_as_an_upright_jpeg_named_jpg():
    data, mime, name = images.for_storage(HEIC, "image/heic", "IMG_1234.HEIC")
    img = _decoded(data)
    assert (img.format, img.size, mime, name) == ("JPEG", (40, 80), "image/jpeg", "IMG_1234.jpg")


def test_a_heic_is_known_by_its_bytes_when_stored():
    data, mime, name = images.for_storage(HEIC, "application/octet-stream", "slip.bin")
    assert (_decoded(data).format, mime, name) == ("JPEG", "image/jpeg", "slip.jpg")


@pytest.mark.parametrize(
    ("fmt", "mime"), [("WEBP", "image/webp"), ("BMP", "image/bmp"), ("TIFF", "image/tiff")]
)
def test_types_xero_refuses_or_that_are_huge_become_jpeg(fmt, mime):
    data, out_mime, name = images.for_storage(_photo(300, 200, fmt=fmt), mime, "scan.x")
    assert (_decoded(data).format, out_mime, name) == ("JPEG", "image/jpeg", "scan.jpg")


def test_a_photo_saved_as_png_becomes_a_smaller_jpeg():
    original = _photo(3000, 2000, fmt="PNG")
    data, mime, name = images.for_storage(original, "image/png", "photo.png")
    assert (_decoded(data).size, mime, name) == ((2048, 1365), "image/jpeg", "photo.jpg")
    assert len(data) < len(original)


def test_a_screenshot_png_that_jpeg_would_not_shrink_stays_png():
    original = _png(1200, 800, colour=(255, 255, 255))
    assert images.for_storage(original, "image/png", "shot.png") == (
        original,
        "image/png",
        "shot.png",
    )


@pytest.mark.parametrize(
    ("data", "mime", "name"),
    [
        (b"%PDF-1.7 pdf bytes", "application/pdf", "a.pdf"),
        (b"GIF89a gif bytes", "image/gif", "a.gif"),
        (b"<svg></svg>", "image/svg+xml", "a.svg"),
        (b"<html></html>", "text/html", "a.html"),
        (b"a,b\n1,2", "text/csv", "a.csv"),
    ],
)
def test_documents_are_stored_as_uploaded(data, mime, name):
    assert images.for_storage(data, mime, name) == (data, mime, name)


@pytest.mark.parametrize("mime", ["image/jpeg", "image/png", "image/heic", "image/webp"])
def test_a_photo_that_does_not_decode_is_refused(mime):
    with pytest.raises(images.ImageError):
        images.for_storage(b"\xff\xd8\xff\xe0 not really a photo", mime, "a.jpg")


def test_a_big_jpeg_is_decoded_at_a_reduced_scale(monkeypatch):
    """The memory guard: a JPEG at least twice the cap is read through draft() (1/2, 1/4, 1/8)."""
    from PIL import JpegImagePlugin

    asked = []
    real = JpegImagePlugin.JpegImageFile.draft

    def spy(self, mode, size):
        asked.append(size)
        return real(self, mode, size)

    monkeypatch.setattr(JpegImagePlugin.JpegImageFile, "draft", spy)
    data, _, _ = images.for_storage(_photo(4400, 3300, quality=90), "image/jpeg", "big.jpg")
    assert asked and _decoded(data).size == (2048, 1536)

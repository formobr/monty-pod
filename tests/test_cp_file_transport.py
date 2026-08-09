"""file:// transport in cp.download/upload — the local backend runs render_spec on the origin/laptop
with no R2 and no CP, handing the pod file:// urls that degrade to a local copy."""
from __future__ import annotations

from pathlib import Path

import pytest

from podagent import cp


class _Response:
    def raise_for_status(self) -> None:
        return None


class _Store:
    def __init__(self) -> None:
        self.headers = None

    def put(self, _url, *, data, headers, timeout):
        self.headers = headers
        return _Response()


def test_download_file_url_copies(tmp_path: Path) -> None:
    src = tmp_path / "src.bin"
    src.write_bytes(b"payload")
    dest = tmp_path / "sub" / "dest.bin"
    cp.download(src.as_uri(), dest)  # file:///abs...
    assert dest.read_bytes() == b"payload"


def test_upload_file_url_copies(tmp_path: Path) -> None:
    src = tmp_path / "out.bin"
    src.write_bytes(b"rendered")
    target = tmp_path / "master" / "final.bin"
    cp.upload(src, target.as_uri())  # parents created
    assert target.read_bytes() == b"rendered"


def test_file_path_none_for_http() -> None:
    assert cp._file_path("https://r2.example/x?sig=1") is None
    assert cp._file_path("http://cp.local/y") is None


def test_snapshot_upload_preserves_browser_image_mime(tmp_path: Path, monkeypatch) -> None:
    """The input-cache snapshot has no image suffix; bytes must still make R2 browser-safe."""
    snapshot = tmp_path / "payload.put-snapshot"
    snapshot.write_bytes(b"\xff\xd8\xff\xe0" + b"jpeg-body")
    store = _Store()
    monkeypatch.setattr(cp, "_store", store)

    cp.upload(snapshot, "https://r2.example/object?signature=redacted")

    assert store.headers["Content-Type"] == "image/jpeg"


def test_unknown_snapshot_remains_binary(tmp_path: Path, monkeypatch) -> None:
    snapshot = tmp_path / "payload.put-snapshot"
    snapshot.write_bytes(b"not a browser image")
    store = _Store()
    monkeypatch.setattr(cp, "_store", store)

    cp.upload(snapshot, "https://r2.example/object?signature=redacted")

    assert store.headers["Content-Type"] == "application/octet-stream"


@pytest.mark.parametrize(("head", "want"), [
    (b"\x89PNG\r\n\x1a\nrest", "image/png"),
    (b"GIF89arest", "image/gif"),
    (b"RIFF\x08\x00\x00\x00WEBPrest", "image/webp"),
    (b"RIFF\x08\x00\x00\x00AVI rest", "application/octet-stream"),
])
def test_snapshot_image_magic_is_closed(tmp_path: Path, head: bytes, want: str) -> None:
    snapshot = tmp_path / "payload.put-snapshot"
    snapshot.write_bytes(head)
    assert cp._upload_content_type(snapshot) == want


def test_explicit_upload_content_type_wins_over_magic(tmp_path: Path, monkeypatch) -> None:
    snapshot = tmp_path / "payload.put-snapshot"
    snapshot.write_bytes(b"\xff\xd8\xff\xe0jpeg-body")
    store = _Store()
    monkeypatch.setattr(cp, "_store", store)

    cp.upload(snapshot, "https://r2.example/object?signature=redacted", "application/custom")

    assert store.headers["Content-Type"] == "application/custom"

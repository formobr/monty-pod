"""file:// transport in cp.download/upload — the local backend runs render_spec on the origin/laptop
with no R2 and no CP, handing the pod file:// urls that degrade to a local copy."""
from __future__ import annotations

from pathlib import Path

from podagent import cp


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

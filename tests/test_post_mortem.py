"""A SECOND BOOT ON ONE RENT IS A DEATH NOBODY WROTE DOWN (`podagent.main.POST_MORTEM_WHY`).

The measured preview run had two `agent up` events against a single rent: the agent died at run-offset
62.7 s and came back at 114.3 s. Fifty-one seconds of silence, twelve envelopes dropped by a control plane
whose socket had gone, and by the time anyone asked why, the container log had died with the pod.

These are NEGATIVE tests. A crash that reads as a clean stop, a clean stop that reads as a crash, and an
unreadable cgroup that reads as "nobody was OOM-killed" are each the instrument lying in the direction that
ends the investigation early.
"""
from __future__ import annotations

import podagent.main as M
import pytest


@pytest.fixture
def mark(tmp_path, monkeypatch):
    p = tmp_path / "podagent.alive"
    monkeypatch.setattr(M, "_LIVE_MARK", p)
    return p


def test_a_first_boot_says_so(mark, monkeypatch):
    monkeypatch.setattr(M, "_oom_kills", lambda: "0")
    assert "prev=none" in M._post_mortem()


def test_a_surviving_mark_is_read_as_a_death(mark, monkeypatch):
    monkeypatch.setattr(M, "_oom_kills", lambda: "0")
    M._mark_alive()
    got = M._post_mortem()
    assert "prev=UNCLEAN" in got, got
    assert "taken, not stopped" in got


def test_a_clean_stop_is_not_reported_as_a_death(mark, monkeypatch):
    monkeypatch.setattr(M, "_oom_kills", lambda: "0")
    M._mark_alive()
    M._mark_stopped()
    assert "prev=none" in M._post_mortem(), "a deliberate stop was mourned as a crash"


def test_the_oom_counter_rides_the_beacon(mark, monkeypatch):
    monkeypatch.setattr(M, "_oom_kills", lambda: "3")
    assert "oom_kills=3" in M._post_mortem()


def test_the_v2_and_v1_shapes_both_parse():
    assert M._oom_from("low 0\nhigh 0\nmax 2\noom 1\noom_kill 4\n") == "4"
    assert M._oom_from("oom_kill_disable 0\nunder_oom 0\noom_kill 0\n") == "0"


def test_a_file_with_no_oom_kill_line_is_not_a_zero():
    """`None` means «this file does not carry the answer», so the caller tries the next cgroup version.
    Returning "0" here would report a kernel that never told us as a kernel that killed nobody."""
    assert M._oom_from("low 0\nhigh 0\nmax 2\n") is None
    assert M._oom_from("oom_kill notanumber\n") is None


def test_an_unreadable_cgroup_is_unknown_and_never_zero(monkeypatch):
    """The instrument may not exonerate the memory budget on evidence it could not read."""
    monkeypatch.setattr(M, "_OOM_FILES", (M.Path("/definitely/not/a/cgroup"),))
    assert M._oom_kills() == "unknown"


def test_a_mark_that_cannot_be_written_is_said_out_loud(monkeypatch, capsys, tmp_path):
    """A silent failure here costs the NEXT post-mortem its only evidence."""
    monkeypatch.setattr(M, "_LIVE_MARK", tmp_path / "nope" / "x" / "podagent.alive")
    monkeypatch.setattr(M.Path, "mkdir", lambda *a, **k: (_ for _ in ()).throw(OSError("read-only fs")))
    M._mark_alive()
    assert "could not write the liveness mark" in capsys.readouterr().err

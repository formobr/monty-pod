"""GATE for prod job 2913bf1a (slug 17-2024): a very quiet source made ffmpeg's loudnorm fall back from
linear to DYNAMIC mode — short of target, TP aim ignored on transients — and the master shipped at +0.41
dBTP. The ceiling is a brickwall now, and what was ENCODED is measured before the file is handed on."""
from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from podagent import finalize

# the 17-2024 shape: +16.2 dB of gain would land the peak at +11.6 dBTP, so linear mode is impossible
QUIET = {"input_i": "-30.8", "input_tp": "-4.6", "input_lra": "2.9", "input_thresh": "-41.0"}
INFEASIBLE = {"input_i": "-20.0", "input_tp": "-2.0", "input_lra": "5.0", "input_thresh": "-30.0"}
# in_tp + (target - in_i) <= aim: loudnorm keeps its linear mode, so the chain stays as it was
FEASIBLE = {"input_i": "-16.0", "input_tp": "-6.0", "input_lra": "6.5", "input_thresh": "-26.4"}
LOUD = {"input_i": "-9.2", "input_tp": "-0.4", "input_lra": "7.2", "input_thresh": "-19.8"}
# clipping-hot mic already AT target: master_af's crackle guard ships it unencoded
HOT = {"input_i": "-15.0", "input_tp": "-0.1", "input_lra": "6.0", "input_thresh": "-25.4"}


def _ln(i: float = -14.0, tp: float = -1.0):
    return SimpleNamespace(i=i, tp=tp, lra=11.0, attenuate_only=False)


def _json(mv: dict) -> str:
    return "{" + ", ".join(f'"{k}" : "{v}"' for k, v in mv.items()) + " }"


def test_a_feasible_linear_gain_keeps_the_loudnorm_chain_and_ends_in_the_brickwall() -> None:
    af, note = finalize.master_af(FEASIBLE, -14.0, -2.2, False)
    assert af == ("loudnorm=I=-14.0:TP=-2.2:LRA=11:linear=true:measured_I=-16.0:measured_TP=-6.0"
                  ":measured_LRA=6.5:measured_thresh=-26.4"
                  ",alimiter=limit=0.7762:attack=5:release=50:level=false")
    assert note.endswith("via linear loudnorm")


def test_an_infeasible_linear_gain_stops_asking_ffmpeg_and_gains_it_itself() -> None:
    """`linear=true` is a REQUEST: ffmpeg downgrades to dynamic whenever in_tp + gain > TP, which lands
    the integrated loudness short (-16.69 on 17-2024). Our own gain + the brickwall cannot downgrade."""
    af, note = finalize.master_af(INFEASIBLE, -14.0, -2.2, False)
    assert af == "volume=6.00dB,alimiter=limit=0.7762:attack=5:release=50:level=false"
    assert "loudnorm" not in af
    assert note == "-20.0 -> -14.0 LUFS (normalize) via volume+limiter (linear infeasible: pred_tp 4.00 > aim -2.2)"


def test_the_branch_is_decided_at_the_aim_not_at_the_ceiling() -> None:
    """ffmpeg downgrades against its own TP argument, which is the AIM — a line drawn at the brand's
    ceiling would keep handing it the -1.8 dBTP prediction it silently turns into dynamic mode."""
    under_aim = {"input_i": "-20.0", "input_tp": "-8.5", "input_lra": "5.0", "input_thresh": "-30.0"}
    assert "loudnorm=" in finalize.master_af(under_aim, -14.0, -2.2, False)[0]
    between = dict(under_aim, input_tp="-7.8")   # pred -1.8: past the aim, still under the -1.0 ceiling
    assert finalize.master_af(between, -14.0, -2.2, False)[0].startswith("volume=")


def test_the_limit_is_the_aim_in_linear_amplitude_not_dB() -> None:
    """alimiter's `limit` is amplitude: a -2.2 dBTP aim written as `-2.2` is silence-with-a-sign, and a
    `1.0` limit is no ceiling at all."""
    assert finalize.limiter_af(-2.2) == "alimiter=limit=0.7762:attack=5:release=50:level=false"
    assert finalize.limiter_af(-2.7) == "alimiter=limit=0.7328:attack=5:release=50:level=false"
    assert finalize.limiter_af(0.0).startswith("alimiter=limit=1.0:")


def test_a_tighter_ceiling_moves_the_limit_with_it(monkeypatch, tmp_path) -> None:
    """The aim is derived (TP - headroom), so the brickwall must follow the brand's TP, not a literal."""
    calls: list[list[str]] = []
    _stub(monkeypatch, tmp_path, calls, FEASIBLE, FEASIBLE)
    finalize.apply_loudnorm(SimpleNamespace(loudnorm=_ln(tp=-1.5)), tmp_path / "m.mp4", tmp_path / "o.mp4")
    apply = calls[1]
    af = apply[apply.index("-af") + 1]
    assert "TP=-2.7:" in af and "alimiter=limit=0.7328:" in af


def test_a_master_already_under_the_aim_is_limited_but_ungained() -> None:
    """The four masters that passed that day must stay gain-identical: level=false makes the brickwall a
    pure ceiling (no makeup, no auto-level), so it is a no-op by construction on signal below `limit`."""
    af, _note = finalize.master_af(LOUD, -14.0, -2.2, False)
    assert "alimiter=" in af
    assert "level=false" in af and "level=true" not in af
    assert not re.search(r"alimiter=[^,]*(makeup|:level=disabled)", af)


def test_the_headroom_still_covers_only_the_codec() -> None:
    assert finalize.TP_HEADROOM_DB == 1.2


# --- the post-encode verdict --------------------------------------------------

def _stub(monkeypatch, tmp_path, calls, first: dict, second: dict) -> None:
    """ffmpeg stand-in: two measure passes (`-f null`) answer `first` then `second`, the apply pass
    writes the output file apply_loudnorm's own size check reads."""
    answers = [first, second]

    def fake(cmd, **_kw):
        calls.append(list(cmd))
        if "null" in cmd:
            return SimpleNamespace(returncode=0, stdout="", stderr=_json(answers.pop(0)))
        (tmp_path / "o.mp4").write_bytes(b"v")
        return SimpleNamespace(returncode=0, stdout="", stderr="")
    monkeypatch.setattr(finalize.subprocess, "run", fake)


def test_the_delivered_master_is_measured_and_its_numbers_are_printed(monkeypatch, tmp_path, capsys) -> None:
    calls: list[list[str]] = []
    _stub(monkeypatch, tmp_path, calls, QUIET,
          {"input_i": "-14.1", "input_tp": "-1.6", "input_lra": "2.8", "input_thresh": "-24.3"})
    out = finalize.apply_loudnorm(SimpleNamespace(loudnorm=_ln()), tmp_path / "m.mp4", tmp_path / "o.mp4")
    assert out == tmp_path / "o.mp4"
    assert len(calls) == 3, "measure -> apply -> verify"
    verify = calls[2]
    assert str(tmp_path / "o.mp4") in verify, "the verdict must measure the ENCODED file, not the source"
    assert verify[verify.index("-map") + 1] == "0:a:0?"
    assert "[finalize] master: delivered lufs=-14.1 tp=-1.6 lra=2.8" in capsys.readouterr().out


def test_a_master_over_the_ceiling_refuses_instead_of_shipping(monkeypatch, tmp_path) -> None:
    """The 2913bf1a shape: the chain ran, the file exists, and it is +0.41 dBTP. Returning it (or the
    un-normalized source) is the silent failure — the pod op must fail loud with the numbers."""
    calls: list[list[str]] = []
    _stub(monkeypatch, tmp_path, calls, QUIET,
          {"input_i": "-16.69", "input_tp": "0.41", "input_lra": "2.9", "input_thresh": "-27.1"})
    with pytest.raises(RuntimeError, match=r"OFF-CONTRACT.*tp=0\.41 ceil=-1\.0"):
        finalize.apply_loudnorm(SimpleNamespace(loudnorm=_ln()), tmp_path / "m.mp4", tmp_path / "o.mp4")


def test_a_master_exactly_at_the_ceiling_ships(monkeypatch, tmp_path) -> None:
    """The ceiling is the contract, the aim is only the margin — refusing at the aim fails masters the
    box accepts."""
    calls: list[list[str]] = []
    _stub(monkeypatch, tmp_path, calls, QUIET,
          {"input_i": "-14.0", "input_tp": "-1.0", "input_lra": "3.0", "input_thresh": "-24.0"})
    assert finalize.apply_loudnorm(SimpleNamespace(loudnorm=_ln()),
                                   tmp_path / "m.mp4", tmp_path / "o.mp4") == tmp_path / "o.mp4"


def test_an_unreadable_verdict_ships_the_master_and_says_so(monkeypatch, tmp_path, capsys) -> None:
    """A measure that cannot be parsed is not evidence of a bad master; the finished render still ships
    (the measure is non-critical by contract), but the log must not read as a pass."""
    def fake(cmd, **_kw):
        if "null" in cmd and str(tmp_path / "o.mp4") in cmd:
            return SimpleNamespace(returncode=1, stdout="", stderr="Output file #0 does not contain")
        if "null" in cmd:
            return SimpleNamespace(returncode=0, stdout="", stderr=_json(QUIET))
        (tmp_path / "o.mp4").write_bytes(b"v")
        return SimpleNamespace(returncode=0, stdout="", stderr="")
    monkeypatch.setattr(finalize.subprocess, "run", fake)
    out = finalize.apply_loudnorm(SimpleNamespace(loudnorm=_ln()), tmp_path / "m.mp4", tmp_path / "o.mp4")
    assert out == tmp_path / "o.mp4"
    assert "UNVERIFIED" in capsys.readouterr().out


def test_a_hot_source_shipped_clean_never_reaches_the_verdict(monkeypatch, tmp_path) -> None:
    """attenuate_only + already at target returns the SOURCE unencoded — measuring an output that was
    never written would turn the crackle guard into a crash."""
    calls: list[list[str]] = []

    def fake(cmd, **_kw):
        calls.append(list(cmd))
        return SimpleNamespace(returncode=0, stdout="", stderr=_json(HOT))
    monkeypatch.setattr(finalize.subprocess, "run", fake)
    fin = SimpleNamespace(loudnorm=SimpleNamespace(i=-14.0, tp=-1.0, lra=11.0, attenuate_only=True))
    src = tmp_path / "m.mp4"
    assert finalize.apply_loudnorm(fin, src, tmp_path / "o.mp4") == src
    assert len(calls) == 1


def test_an_unmeasurable_source_is_still_left_at_source_level(monkeypatch, tmp_path, capsys) -> None:
    """Unchanged contract for the FIRST pass: it must not become the new hard failure."""
    monkeypatch.setattr(finalize.subprocess, "run",
                        lambda *_a, **_k: SimpleNamespace(returncode=1, stdout="", stderr="no json here"))
    src = tmp_path / "m.mp4"
    assert finalize.apply_loudnorm(SimpleNamespace(loudnorm=_ln()), src, tmp_path / "o.mp4") == src
    assert "left at source level" in capsys.readouterr().out


@pytest.mark.integration
def test_the_real_17_2024_shape_is_delivered_on_target_and_under_the_ceiling(tmp_path: Path) -> None:
    """The regression itself, end to end: a source measured -30.6 LUFS / -6.7 dBTP (the prod one was
    -30.8 dB mean, -4.6 dB max) is exactly what ffmpeg answers with dynamic mode, and argv assertions
    can neither see that nor tell a valid filtergraph from a typo."""
    import shutil
    import subprocess
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg unavailable")
    src = tmp_path / "quiet.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
         "-f", "lavfi", "-i", "testsrc=size=64x64:rate=10",
         "-f", "lavfi", "-i", "anoisesrc=c=pink:r=48000:d=6:a=0.15",
         # eval=frame gates on ~21 ms audio frames, so a literal 5 ms window would land between two
         "-af", "volume='if(between(t,1.5,1.53)+between(t,3.5,3.53),4.8,1.0)':eval=frame",
         "-map", "0:v", "-map", "1:a", "-t", "6", "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-b:a", "192k", "-shortest", str(src)], check=True)
    ln = _ln()
    # apply_loudnorm RAISES over the ceiling, so returning the encoded path is already half the verdict
    out = finalize.apply_loudnorm(SimpleNamespace(loudnorm=ln), src, tmp_path / "o.mp4")
    assert out == tmp_path / "o.mp4" and out.stat().st_size > 0

    meas = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostats", "-i", str(out), "-map", "0:a:0?",
         "-af", "loudnorm=I=-14:TP=-2.2:LRA=11:print_format=json", "-f", "null", "-"],
        capture_output=True, text=True)
    got = re.search(r"\{[^{}]*\"input_i\"[^{}]*\}", meas.stderr, re.S)
    assert got, meas.stderr[-2000:]
    delivered = {k: float(v) for k, v in re.findall(r'"(input_i|input_tp)"\s*:\s*"(-?[\d.]+)"', got.group(0))}
    assert delivered["input_i"] == pytest.approx(ln.i, abs=3.0), delivered
    assert delivered["input_tp"] <= ln.tp, delivered

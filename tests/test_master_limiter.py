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
# on the brand's band: the answer every measure gets when the pass under test is not the one being judged
ON_BAND = {"input_i": "-14.1", "input_tp": "-1.6", "input_lra": "3.0", "input_thresh": "-24.1"}
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
    _stub(monkeypatch, tmp_path, calls, FEASIBLE, ON_BAND, ON_BAND)
    finalize.apply_loudnorm(SimpleNamespace(loudnorm=_ln(tp=-1.5)), tmp_path / "m.mp4", tmp_path / "o.mp4")
    af = _afs(calls)[0]
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

def _stub(monkeypatch, tmp_path, calls, *answers: dict) -> None:
    """ffmpeg stand-in: the measure passes (`-f null`) answer `answers` in order — source, levelled PCM,
    [made-up PCM], delivered master — and every other pass writes the file it names."""
    left = list(answers)

    def fake(cmd, **_kw):
        calls.append(list(cmd))
        if "null" in cmd:
            return SimpleNamespace(returncode=0, stdout="", stderr=_json(left.pop(0)))
        Path(cmd[-2]).write_bytes(b"v")
        return SimpleNamespace(returncode=0, stdout="", stderr="")
    monkeypatch.setattr(finalize.subprocess, "run", fake)


def _afs(calls) -> list[str]:
    return [c[c.index("-af") + 1] for c in calls if "-af" in c and "null" not in c]


def test_the_delivered_master_is_measured_and_its_numbers_are_printed(monkeypatch, tmp_path, capsys) -> None:
    calls: list[list[str]] = []
    _stub(monkeypatch, tmp_path, calls, QUIET, ON_BAND,
          {"input_i": "-14.1", "input_tp": "-1.6", "input_lra": "2.8", "input_thresh": "-24.3"})
    out = finalize.apply_loudnorm(SimpleNamespace(loudnorm=_ln()), tmp_path / "m.mp4", tmp_path / "o.mp4")
    assert out == tmp_path / "o.mp4"
    assert len(calls) == 5, "measure -> level PCM -> measure PCM -> mux -> verify"
    verify = calls[4]
    assert str(tmp_path / "o.mp4") in verify, "the verdict must measure the ENCODED file, not the source"
    assert verify[verify.index("-map") + 1] == "0:a:0?"
    assert "[finalize] master: delivered lufs=-14.1 tp=-1.6 lra=2.8" in capsys.readouterr().out


def test_the_level_work_happens_in_pcm_before_the_single_aac_encode(monkeypatch, tmp_path) -> None:
    """Measuring through the codec is why the make-up could never see what the brickwall removed; one
    encode also stays one encode — a second aac generation on a finished master is not free."""
    calls: list[list[str]] = []
    _stub(monkeypatch, tmp_path, calls, QUIET, ON_BAND, ON_BAND)
    finalize.apply_loudnorm(SimpleNamespace(loudnorm=_ln()), tmp_path / "m.mp4", tmp_path / "o.mp4")
    level, mux = calls[1], calls[3]
    assert "-vn" in level and level[level.index("-c:a") + 1] == "pcm_s16le"
    assert str(tmp_path / "o.lvl.wav") == level[-2]
    assert mux[mux.index("-c:a") + 1] == "aac" and "copy" == mux[mux.index("-c:v") + 1]
    assert len([c for c in calls if "-c:a" in c and c[c.index("-c:a") + 1] == "aac"]) == 1
    assert not (tmp_path / "o.lvl.wav").exists(), "the intermediate PCM must not survive the job"


def test_a_master_over_the_ceiling_refuses_instead_of_shipping(monkeypatch, tmp_path) -> None:
    """The 2913bf1a shape: the chain ran, the file exists, and it is +0.41 dBTP. Returning it (or the
    un-normalized source) is the silent failure — the pod op must fail loud with the numbers."""
    calls: list[list[str]] = []
    _stub(monkeypatch, tmp_path, calls, QUIET, ON_BAND,
          {"input_i": "-16.69", "input_tp": "0.41", "input_lra": "2.9", "input_thresh": "-27.1"})
    with pytest.raises(RuntimeError, match=r"OFF-CONTRACT.*tp=0\.41 ceil=-1\.0"):
        finalize.apply_loudnorm(SimpleNamespace(loudnorm=_ln()), tmp_path / "m.mp4", tmp_path / "o.mp4")


def test_a_delivered_master_under_the_bands_floor_refuses_too(monkeypatch, tmp_path) -> None:
    """The other half of 2913bf1a: -16.69 LUFS is off-contract even at a clean true peak, and the AAC
    encode is the last place the level can still move after the make-up."""
    calls: list[list[str]] = []
    _stub(monkeypatch, tmp_path, calls, QUIET, ON_BAND,
          {"input_i": "-17.4", "input_tp": "-1.9", "input_lra": "2.9", "input_thresh": "-27.1"})
    with pytest.raises(RuntimeError, match=r"OFF-CONTRACT.*lufs=-17\.4 target=-14\.0 tol=3\.0"):
        finalize.apply_loudnorm(SimpleNamespace(loudnorm=_ln()), tmp_path / "m.mp4", tmp_path / "o.mp4")


def test_a_delivered_master_over_the_bands_ceiling_refuses_as_well(monkeypatch, tmp_path) -> None:
    """The make-up can overshoot (its gain is measured on PCM, the aac encode moves the level again), and
    a band checked in one direction only ships a master the box then throws back."""
    calls: list[list[str]] = []
    _stub(monkeypatch, tmp_path, calls, QUIET, ON_BAND,
          {"input_i": "-10.4", "input_tp": "-1.2", "input_lra": "2.9", "input_thresh": "-20.4"})
    with pytest.raises(RuntimeError, match=r"OFF-CONTRACT.*lufs=-10\.4 target=-14\.0 tol=3\.0"):
        finalize.apply_loudnorm(SimpleNamespace(loudnorm=_ln()), tmp_path / "m.mp4", tmp_path / "o.mp4")


def test_a_refused_master_is_not_left_at_the_delivery_address(monkeypatch, tmp_path) -> None:
    """The caller PUTs whatever sits at that path; a refusal that leaves the file behind is the silent
    ship this whole verdict exists to prevent."""
    calls: list[list[str]] = []
    _stub(monkeypatch, tmp_path, calls, QUIET, ON_BAND,
          {"input_i": "-16.69", "input_tp": "0.41", "input_lra": "2.9", "input_thresh": "-27.1"})
    with pytest.raises(RuntimeError):
        finalize.apply_loudnorm(SimpleNamespace(loudnorm=_ln()), tmp_path / "m.mp4", tmp_path / "o.mp4")
    assert not (tmp_path / "o.mp4").exists()
    assert not (tmp_path / "o.lvl.wav").exists() and not (tmp_path / "o.mk.wav").exists()


def test_a_master_exactly_at_the_ceiling_ships(monkeypatch, tmp_path) -> None:
    """The ceiling is the contract, the aim is only the margin — refusing at the aim fails masters the
    box accepts."""
    calls: list[list[str]] = []
    _stub(monkeypatch, tmp_path, calls, QUIET, ON_BAND,
          {"input_i": "-14.0", "input_tp": "-1.0", "input_lra": "3.0", "input_thresh": "-24.0"})
    assert finalize.apply_loudnorm(SimpleNamespace(loudnorm=_ln()),
                                   tmp_path / "m.mp4", tmp_path / "o.mp4") == tmp_path / "o.mp4"


# --- the make-up: what the brickwall removed, measured and given back ---------

def test_a_short_levelled_master_gets_the_residual_back_under_the_same_ceiling(
        monkeypatch, tmp_path, capsys) -> None:
    """The gain is computed from the SOURCE measure, so a brickwall that removes transient-carried
    loudness leaves the master short — 17-2024 landed -16.69 exactly this way."""
    calls: list[list[str]] = []
    _stub(monkeypatch, tmp_path, calls, QUIET,
          {"input_i": "-17.0", "input_tp": "-2.2", "input_lra": "2.9", "input_thresh": "-27.0"},
          {"input_i": "-14.2", "input_tp": "-2.2", "input_lra": "3.1", "input_thresh": "-24.2"},
          {"input_i": "-14.2", "input_tp": "-1.5", "input_lra": "3.1", "input_thresh": "-24.2"})
    out = finalize.apply_loudnorm(SimpleNamespace(loudnorm=_ln()), tmp_path / "m.mp4", tmp_path / "o.mp4")
    assert out == tmp_path / "o.mp4"
    assert _afs(calls)[1] == "volume=3.00dB,alimiter=limit=0.7762:attack=5:release=50:level=false"
    log = capsys.readouterr().out
    assert "levelled lufs=-17.0 tp=-2.2 (residual +3.00 LU)" in log
    assert "make-up +3.00 dB -> lufs=-14.2 tp=-2.2" in log


def test_a_master_already_on_target_gets_no_second_pass(monkeypatch, tmp_path, capsys) -> None:
    """A make-up pass is another decode/encode of the whole master and another trip through the
    limiter; inside the band there is nothing to buy with it."""
    calls: list[list[str]] = []
    _stub(monkeypatch, tmp_path, calls, QUIET,
          {"input_i": "-14.4", "input_tp": "-2.3", "input_lra": "3.0", "input_thresh": "-24.4"}, ON_BAND)
    finalize.apply_loudnorm(SimpleNamespace(loudnorm=_ln()), tmp_path / "m.mp4", tmp_path / "o.mp4")
    assert len(_afs(calls)) == 1, "the level pass only"
    assert "make-up" not in capsys.readouterr().out


def test_loudness_that_lives_only_in_transients_refuses_instead_of_chasing_it(
        monkeypatch, tmp_path) -> None:
    """Corpus `quiet_speech_clicks`: the integrated measure was carried by clicks the ceiling removes, so
    the residual is unbounded — giving it back would only feed the limiter the same transients again."""
    calls: list[list[str]] = []
    _stub(monkeypatch, tmp_path, calls, QUIET,
          {"input_i": "-26.33", "input_tp": "-2.2", "input_lra": "0.4", "input_thresh": "-36.3"})
    with pytest.raises(RuntimeError, match=r"lufs=-26\.33 target=-14\.0 \(residual \+12\.33 LU over the 12\.0 dB"):
        finalize.apply_loudnorm(SimpleNamespace(loudnorm=_ln()), tmp_path / "m.mp4", tmp_path / "o.mp4")
    assert not any("aac" in c for c in calls), "the refusal must come before the master is encoded"
    assert not (tmp_path / "o.lvl.wav").exists()


def test_an_unreadable_verdict_ships_the_master_and_says_so(monkeypatch, tmp_path, capsys) -> None:
    """A measure that cannot be parsed is not evidence of a bad master; the finished render still ships
    (the measure is non-critical by contract), but the log must not read as a pass."""
    def fake(cmd, **_kw):
        if "null" in cmd and str(tmp_path / "o.mp4") in cmd:
            return SimpleNamespace(returncode=1, stdout="", stderr="Output file #0 does not contain")
        if "null" in cmd:
            return SimpleNamespace(returncode=0, stdout="", stderr=_json(ON_BAND))
        Path(cmd[-2]).write_bytes(b"v")
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


def test_a_hot_source_far_under_target_is_levelled_not_shipped_as_is(monkeypatch, tmp_path) -> None:
    """The crackle guard is a shortcut past the level chain, and past the band with it: at -20 LUFS the
    box refuses the very file the guard called done, so the two ends disagree about one master."""
    calls: list[list[str]] = []
    quiet_hot = {"input_i": "-20.0", "input_tp": "-0.2", "input_lra": "6.0", "input_thresh": "-30.4"}
    _stub(monkeypatch, tmp_path, calls, quiet_hot, ON_BAND, ON_BAND)
    fin = SimpleNamespace(loudnorm=SimpleNamespace(i=-14.0, tp=-1.0, lra=11.0, attenuate_only=True))
    assert finalize.apply_loudnorm(fin, tmp_path / "m.mp4", tmp_path / "o.mp4") == tmp_path / "o.mp4"
    assert _afs(calls)[0].startswith("volume=6.00dB,alimiter=")


def test_the_band_edges_still_take_the_crackle_guards_shortcut() -> None:
    """Inside the band a hot mic must NOT be touched — boosting it is what amplifies the crackle."""
    for i in ("-14.0", "-17.0", "-15.5"):
        af, note = finalize.master_af(dict(HOT, input_i=i), -14.0, -2.2, True)
        assert af is None and "shipped clean" in note, i
    af, _note = finalize.master_af(dict(HOT, input_i="-17.01"), -14.0, -2.2, True)
    assert af is not None, "past the floor the level chain owns it"


def test_every_ffmpeg_wait_names_its_step_when_it_wedges(monkeypatch, tmp_path) -> None:
    """A bare TimeoutExpired out of a finalize pass says nothing about WHICH pass wedged; the pod's error
    is all the box gets."""
    def boom(cmd, **kw):
        assert kw.get("timeout") == finalize._AUDIO_PASS_WALL_S, cmd
        raise finalize.subprocess.TimeoutExpired(cmd, kw["timeout"])
    monkeypatch.setattr(finalize.subprocess, "run", boom)
    with pytest.raises(RuntimeError, match="master measure ffmpeg timed out after 300s"):
        finalize.apply_loudnorm(SimpleNamespace(loudnorm=_ln()), tmp_path / "m.mp4", tmp_path / "o.mp4")

    calls: list[list[str]] = []

    def wedge_level(cmd, **kw):
        calls.append(list(cmd))
        if "null" in cmd:
            return SimpleNamespace(returncode=0, stdout="", stderr=_json(QUIET))
        raise finalize.subprocess.TimeoutExpired(cmd, kw["timeout"])
    monkeypatch.setattr(finalize.subprocess, "run", wedge_level)
    with pytest.raises(RuntimeError, match="master level ffmpeg timed out after 300s"):
        finalize.apply_loudnorm(SimpleNamespace(loudnorm=_ln()), tmp_path / "m.mp4", tmp_path / "o.mp4")
    assert not (tmp_path / "o.lvl.wav").exists(), "a wedged pass still leaves no PCM behind"


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

    delivered = _delivered(out)
    # tighter than the box's own +-3 band: this shape has to land ON target, not merely inside it
    assert delivered["input_i"] == pytest.approx(ln.i, abs=1.0), delivered
    assert delivered["input_tp"] <= ln.tp, delivered


@pytest.mark.integration
def test_a_click_carried_source_is_either_on_band_or_refused_with_its_numbers(tmp_path: Path) -> None:
    """Corpus `quiet_speech_clicks` shape at a speech-level bed: 5 ms clicks at -4 dBFS every 1.5 s over a
    -30 dBFS bed. The clicks carry part of the integrated measure and the ceiling then removes them, which
    is what the make-up pass exists to answer — and if it cannot, the numbers must be in the refusal."""
    import shutil
    import subprocess
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg unavailable")
    clicks = "+".join(rf"between(t\,{t}\,{t + 0.005})" for t in (1.5, 3.0, 4.5, 6.0, 7.5))
    src = tmp_path / "clicks.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
         "-f", "lavfi", "-i", "testsrc2=s=320x240:r=30:d=8",
         "-f", "lavfi", "-i", "anoisesrc=r=48000:c=pink:a=0.15:d=8:seed=17",
         "-f", "lavfi", "-i", rf"aevalsrc=0.63*sin(2*PI*1000*t)*({clicks}):d=8:s=48000",
         "-filter_complex", "[1:a][2:a]amix=inputs=2:duration=first:normalize=0,"
                            "aformat=channel_layouts=stereo:sample_rates=48000[a]",
         "-map", "0:v", "-map", "[a]", "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-b:a", "192k", "-shortest", str(src)], check=True)
    ln = _ln()
    try:
        out = finalize.apply_loudnorm(SimpleNamespace(loudnorm=ln), src, tmp_path / "o.mp4")
    except RuntimeError as exc:
        assert re.search(r"lufs=-\d+\.\d+ target=-14\.0 \(residual \+\d+\.\d+ LU", str(exc)), exc
        return
    delivered = _delivered(out)
    assert delivered["input_i"] >= -17.0, delivered
    assert delivered["input_tp"] <= ln.tp, delivered


def _delivered(path: Path) -> dict[str, float]:
    """The same measure the box's check_master takes, so the number asserted is the number it would see."""
    import subprocess
    meas = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostats", "-i", str(path), "-map", "0:a:0?",
         "-af", "loudnorm=I=-14:TP=-2.2:LRA=11:print_format=json", "-f", "null", "-"],
        capture_output=True, text=True)
    got = re.search(r"\{[^{}]*\"input_i\"[^{}]*\}", meas.stderr, re.S)
    assert got, meas.stderr[-2000:]
    return {k: float(v) for k, v in re.findall(r'"(input_i|input_tp)"\s*:\s*"(-?[\d.]+)"', got.group(0))}

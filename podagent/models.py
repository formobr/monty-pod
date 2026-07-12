"""Pydantic mirror of contracts/ (the SSOT JSON Schemas). Kept in lockstep by
tests/test_contracts_goldens.py — every golden must round-trip through both."""
from __future__ import annotations

from typing import Annotated, Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SPEC_VERSION: Final = 1


class SpecInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    kind: Literal["video", "audio", "image", "font", "code"]
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    url: str = Field(min_length=1)


class TimelineSegment(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    src: str = Field(min_length=1)
    in_: float = Field(ge=0, alias="in")
    out: float = Field(gt=0)
    speed: float = Field(gt=0)

    @model_validator(mode="after")
    def _out_after_in(self) -> "TimelineSegment":
        if self.out <= self.in_:
            raise ValueError(f"segment out={self.out} must be > in={self.in_}")
        return self


class Timeline(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fps: float = Field(gt=0)
    width: int = Field(ge=2)
    height: int = Field(ge=2)
    segments: list[TimelineSegment] = Field(min_length=1)


RectT = Annotated[list[Annotated[float, Field(ge=0, le=1)]], Field(min_length=4, max_length=4)]


class MotionKeyframe(BaseModel):
    model_config = ConfigDict(extra="forbid")

    t: float = Field(ge=0)
    rect: RectT


class MotionSegment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seg: int = Field(ge=0)
    interp: Literal["linear", "ease_in", "ease_out", "ease_in_out"]
    keyframes: list[MotionKeyframe] = Field(min_length=1)

    @model_validator(mode="after")
    def _t_monotone(self) -> "MotionSegment":
        ts = [k.t for k in self.keyframes]
        if any(b < a for a, b in zip(ts, ts[1:])):
            raise ValueError(f"keyframe times must be non-decreasing, got {ts}")
        return self


class Motion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    segments: list[MotionSegment]


class SpecTransition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["slide_wipe", "push", "dissolve"]
    edge: Literal["entry", "return"]
    direction: Literal["left", "right", "up", "down"] | None = None
    dur: float = Field(gt=0)

    @model_validator(mode="after")
    def _direction_matches_kind(self) -> "SpecTransition":
        if self.kind == "dissolve":
            if self.direction is not None:
                raise ValueError("dissolve must not carry a direction")
        elif self.direction is None:
            raise ValueError(f"{self.kind} requires a direction")
        return self


class SpecBrollClip(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    clip: str = Field(min_length=1)
    start: float = Field(ge=0)
    preset: str = Field(min_length=1)
    dur: float | None = Field(default=None, gt=0)
    in_: float | None = Field(default=None, ge=0, alias="in")
    scale: str | None = None
    amount: float | None = None
    accent: bool | None = None
    transition_in: SpecTransition | None = None
    transition_out: SpecTransition | None = None


class SpecBrollFinal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    broll: list[SpecBrollClip]


class SpecMotionSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    comp: str = Field(min_length=1)
    start: float = Field(ge=0)
    props: dict[str, Any]
    glass: bool = False


class SpecCaptionWord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1)
    start: float = Field(ge=0)
    end: float = Field(ge=0)
    hot: bool = False


class SpecCaptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    centerY: float | None = None
    accent: str | None = None
    style: Literal["oneword", "phrase", "phrase_jump"] | None = None
    hot: list[float] = Field(default_factory=list)
    words: list[SpecCaptionWord] = Field(default_factory=list)
    font: str | None = None


class SpecBrandManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tokens: dict[str, Any]
    fonts: dict[str, Any]


class SpecMotionPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sections: list[SpecMotionSection]
    captions: SpecCaptions | None = None
    brand: SpecBrandManifest | None = None


class SpecMusic(BaseModel):
    model_config = ConfigDict(extra="forbid")

    track: str = Field(min_length=1)
    start: float = Field(default=0.0, ge=0)
    gain: float = Field(gt=0, le=2)


class SpecCoverHeadline(BaseModel):
    model_config = ConfigDict(extra="allow")

    lines: list[Any]
    pos: str | None = None
    y: float | None = None
    size: float | None = None
    box: bool | None = None


class SpecCover(BaseModel):
    model_config = ConfigDict(extra="forbid")

    frame_at: float = Field(ge=0)
    headline: SpecCoverHeadline
    logo: dict[str, Any] | None = None
    elements: list[dict[str, Any]] = Field(default_factory=list)
    colors: dict[str, list[int]] | None = None
    font: str | None = None
    display_weight: int = 800


class SpecTrim(BaseModel):
    model_config = ConfigDict(extra="forbid")

    a: float = Field(ge=0)
    b: float = Field(gt=0)

    @model_validator(mode="after")
    def _b_after_a(self) -> "SpecTrim":
        if self.b <= self.a:
            raise ValueError(f"trim end b={self.b} must be > start a={self.a}")
        return self


class SpecSfx(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sound: str = Field(min_length=1)
    at: float = Field(ge=0)
    gain: float = Field(gt=0, le=2)


class Overlays(BaseModel):
    model_config = ConfigDict(extra="forbid")

    broll_final: SpecBrollFinal | None = None
    motion_plan: SpecMotionPlan | None = None
    music: SpecMusic | None = None
    cover: SpecCover | None = None
    trims: list[SpecTrim] | None = None
    sfx: list[SpecSfx] | None = None


class Encode(BaseModel):
    model_config = ConfigDict(extra="forbid")

    video: Literal["h264_nvenc", "libx264"]
    preset: str = Field(min_length=1)
    cq: int = Field(ge=0, le=51)
    pix_fmt: Literal["yuv420p"]
    audio: Literal["aac"]
    audio_bitrate: str = Field(pattern=r"^[0-9]+k$")


class SpecOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    kind: Literal["proxy", "master", "cache", "cover"]   # cover = the standalone cover.png deliverable
    put_url: str = Field(min_length=1)


class RenderSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    spec_version: Literal[1]
    job_id: str = Field(min_length=1)
    slug: str = Field(min_length=1)
    mode: Literal["preview", "final"]
    inputs: list[SpecInput] = Field(min_length=1)
    timeline: Timeline
    motion: Motion | None = None
    overlays: Overlays | None = None
    encode: Encode
    outputs: list[SpecOutput] = Field(min_length=1)
    base_voice_rescued: bool = False

    @model_validator(mode="after")
    def _seam_invariants(self) -> "RenderSpec":
        if self.mode == "preview" and self.overlays is not None:
            raise ValueError("preview carries no overlays")

        ids = {i.id for i in self.inputs}
        if len(ids) != len(self.inputs):
            raise ValueError("inputs[].id must be unique")
        for n, seg in enumerate(self.timeline.segments):
            if seg.src not in ids:
                raise ValueError(f"timeline.segments[{n}].src {seg.src!r} is not an inputs[].id")
        if self.overlays is not None:
            if self.overlays.broll_final is not None:
                for n, c in enumerate(self.overlays.broll_final.broll):
                    if c.clip not in ids:
                        raise ValueError(f"broll[{n}].clip {c.clip!r} is not an inputs[].id")
            for n, s in enumerate(self.overlays.sfx or []):
                if s.sound not in ids:
                    raise ValueError(f"sfx[{n}].sound {s.sound!r} is not an inputs[].id")
            if self.overlays.music is not None and self.overlays.music.track not in ids:
                raise ValueError(f"music.track {self.overlays.music.track!r} is not an inputs[].id")

        if self.motion is not None:
            nseg = len(self.timeline.segments)
            for m in self.motion.segments:
                if m.seg >= nseg:
                    raise ValueError(f"motion seg={m.seg} out of range ({nseg} segments)")
        return self


SpanT = Annotated[list[Annotated[float, Field(ge=0)]], Field(min_length=2, max_length=2)]


def _spans_ok(spans: list[list[float]], what: str) -> None:
    for n, (a, b) in enumerate((s[0], s[1]) for s in spans):
        if b <= a:
            raise ValueError(f"{what}[{n}] end {b} must be > start {a}")


class AlignParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    audio_url: str = Field(min_length=1)
    windows: list[SpanT] = Field(min_length=1)

    @model_validator(mode="after")
    def _windows_ok(self) -> "AlignParams":
        _spans_ok(self.windows, "windows")
        return self


class FaceProbeParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    video_url: str = Field(min_length=1)
    shots: list[SpanT] = Field(min_length=1)
    stride: int = Field(ge=1)
    frame_diff: bool

    @model_validator(mode="after")
    def _shots_ok(self) -> "FaceProbeParams":
        _spans_ok(self.shots, "shots")
        return self


class InferRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    infer_version: Literal[1]
    job_id: str = Field(min_length=1)
    kind: Literal["align", "face_probe"]
    model: str = Field(min_length=1)
    put_url: str = Field(min_length=1)
    align: AlignParams | None = None
    face_probe: FaceProbeParams | None = None

    @model_validator(mode="after")
    def _block_matches_kind(self) -> "InferRequest":
        want, other = (self.align, self.face_probe) if self.kind == "align" else (self.face_probe, self.align)
        if want is None:
            raise ValueError(f"kind={self.kind} requires its params block")
        if other is not None:
            raise ValueError(f"kind={self.kind} must not carry the other kind's params block")
        return self


class InferTiming(BaseModel):
    model_config = ConfigDict(extra="forbid")

    infer_s: float = Field(ge=0)
    boot_s: float | None = Field(default=None, ge=0)


class InferResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    infer_version: Literal[1]
    job_id: str = Field(min_length=1)
    kind: Literal["align", "face_probe"]
    status: Literal["ok", "error"]
    result_key: str | None = Field(default=None, min_length=1)
    error: str | None = Field(default=None, min_length=1)
    timing: InferTiming | None = None

    @model_validator(mode="after")
    def _status_shape(self) -> "InferResult":
        if self.status == "ok":
            if self.result_key is None or self.timing is None:
                raise ValueError("status=ok requires result_key and timing")
            if self.error is not None:
                raise ValueError("status=ok must not carry error")
        else:
            if self.error is None:
                raise ValueError("status=error requires error")
            if self.result_key is not None:
                raise ValueError("status=error must not carry result_key")
        return self


Box5T = Annotated[list[float], Field(min_length=5, max_length=5)]


class ProbeFrame(BaseModel):
    model_config = ConfigDict(extra="forbid")

    t: float = Field(ge=0)
    boxes: list[Box5T]
    diff: float | None = Field(default=None, ge=0)


class ProbeShot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    a: float = Field(ge=0)
    b: float = Field(gt=0)
    frames: list[ProbeFrame]


class PodJob(BaseModel):
    """The control-plane→pod job envelope (GET /pod/job). No version const of its
    own — request/spec each pin their own version; contracts/VERSION is the shared pin."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["infer", "render"]
    request: InferRequest | None = None
    spec: RenderSpec | None = None

    @model_validator(mode="after")
    def _block_matches_type(self) -> "PodJob":
        want, other = (self.request, self.spec) if self.type == "infer" else (self.spec, self.request)
        if want is None:
            raise ValueError(f"type={self.type} requires its matching block")
        if other is not None:
            raise ValueError(f"type={self.type} must not carry the other type's block")
        return self


class FaceProbePayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str = Field(min_length=1)
    width: int = Field(ge=2)
    height: int = Field(ge=2)
    shots: list[ProbeShot] = Field(min_length=1)

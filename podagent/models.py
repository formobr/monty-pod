"""Pydantic mirror of contracts/ (the SSOT JSON Schemas). Kept in lockstep by
tests/test_contracts_goldens.py — every golden must round-trip through both."""
from __future__ import annotations

import math
from typing import Annotated, Any, Final, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, model_validator

SPEC_VERSION: Final = 7


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
    style: Literal["oneword", "phrase", "phrase_jump", "bold", "word_accent"] | None = None
    hot: list[float] = Field(default_factory=list)
    words: list[SpecCaptionWord] = Field(default_factory=list)
    font: str | None = None
    # mirrors the engine's spec models — the pod's libass burn does not consume it (no Remotion Captions path
    # here), but extra="forbid" means the field must exist or a spec that carries it fails validation.
    maxSeconds: float | None = Field(default=None, gt=0)


class SpecBrandManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tokens: dict[str, Any]
    fonts: dict[str, Any]
    looks: dict[str, Any] | None = None


class BundleRef(BaseModel):
    """Where the pod gets the Remotion bundle this job renders its mograph with. A presigned GET for a tar
    of the project (src + node_modules + render_batch.mjs), plus that tar's sha256 — BOTH the integrity
    check and the cache key, so a rebuilt bundle can never be served from a stale entry."""

    model_config = ConfigDict(extra="forbid")

    url: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size: int | None = Field(default=None, gt=0)


class SpecMotionPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sections: list[SpecMotionSection]
    captions: SpecCaptions | None = None
    brand: SpecBrandManifest | None = None
    bundle: BundleRef | None = None

    @model_validator(mode="after")
    def _sections_need_a_bundle(self) -> "SpecMotionPlan":
        # Nothing renders a section without the bundle, and the image deliberately bakes no copy. Refusing
        # HERE makes a bundle-less job fail at validation with one clear line; without it the pod would get
        # as far as the renderer and then either crash mid-encode or — worse, and the reason this exists —
        # skip the sections and publish a green manifest for a video with no mograph in it.
        if self.sections and self.bundle is None:
            raise ValueError(
                "motion_plan.sections requires motion_plan.bundle — the pod bakes no Remotion bundle, so "
                "sections without one cannot render (see the engine's pod runbook)")
        return self


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


# -- finalize: the delivery tail (final tier only) ------------------------------------------------


class SpecAccent(BaseModel):
    """One resolved frame-accent: WHICH treatment, WHERE on the final clock, HOW hard. The Director's
    reasoning (the anchor phrase, the score, why this beat earned an accent) never crosses — only
    these three resolved values do. `film_burn` is the one two-input kind: `burn`/`clicks` are inputs[].id
    of the burn overlay video and its click sound (mirrors overlays.opener's junction assets, body-side) —
    required together on film_burn, forbidden on every other kind."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["camera_shake", "grain", "zoom_punch", "glitch", "zoom_blur", "rgb_split", "pixelate",
                  "film_burn"]
    at: float = Field(ge=0)
    intensity: float = Field(ge=0, le=1)
    burn: str | None = Field(default=None, min_length=1)
    clicks: str | None = Field(default=None, min_length=1)
    # film_burn only: the engine's precomputed frame slip as [(t_rel, dy_px)] around `at` (the engine's
    # burn-slip keyframes, the SAME list the preview plays). Absent → add_offset_jump draws it itself.
    slip: list[tuple[float, float]] | None = Field(default=None, min_length=2)

    @model_validator(mode="after")
    def _film_burn_needs_both(self) -> "SpecAccent":
        # burn/clicks are typed as plain `string` in the schema (no null in the union) — an EXPLICIT null
        # is therefore as invalid as an unresolvable ref would be, and `model_fields_set` is the only way
        # to see "the key was present" separately from "the value happens to be None" (the schema's
        # `required` inside its film_burn/else allOf branches checks presence, not value).
        burn_set, clicks_set = "burn" in self.model_fields_set, "clicks" in self.model_fields_set
        if burn_set and self.burn is None:
            raise ValueError("accent.burn must not be null — omit the key on kinds that don't carry it")
        if clicks_set and self.clicks is None:
            raise ValueError("accent.clicks must not be null — omit the key on kinds that don't carry it")
        if self.kind == "film_burn":
            if self.burn is None or self.clicks is None:
                raise ValueError("film_burn accent requires both burn and clicks input ids")
        elif burn_set or clicks_set:
            raise ValueError(f"{self.kind} accent must not carry burn/clicks (film_burn only)")
        if self.slip is not None and self.kind != "film_burn":
            raise ValueError(f"{self.kind} accent must not carry slip (film_burn only)")
        return self


class SpecOpener(BaseModel):
    """The pre-rendered cold-open clip and its film-burn junction assets, welded onto the front of the
    body (single-pass prep — E-W1; the pod does not execute it until render_onepass lands, W2). `cold`
    is an inputs[].id (mirrors the engine's cold-open stitcher's own `cold` arg). `burn`/`clicks` are a PAIR — set
    both (the film-burn junction, mirrors the engine's cold-open stitcher's own `must_exist=True` on both) or neither
    (a hard-cut concat) — same style as the accent's two-input kind: plain strings, absence (never an
    explicit null) means "not carried". `cold_trim` shortens the cold-open's tail (seconds); `gain` is
    the junction click's linear level."""

    model_config = ConfigDict(extra="forbid")

    cold: str = Field(min_length=1)
    burn: str | None = Field(default=None, min_length=1)
    clicks: str | None = Field(default=None, min_length=1)
    cold_trim: float = Field(default=0.0, ge=0)
    gain: float = Field(default=0.4, ge=0, le=1)

    @model_validator(mode="after")
    def _junction_pair_or_neither(self) -> "SpecOpener":
        burn_set, clicks_set = "burn" in self.model_fields_set, "clicks" in self.model_fields_set
        if burn_set and self.burn is None:
            raise ValueError("opener.burn must not be null — omit the key entirely for a hard-cut opener")
        if clicks_set and self.clicks is None:
            raise ValueError("opener.clicks must not be null — omit the key entirely for a hard-cut opener")
        if burn_set != clicks_set:
            raise ValueError(
                "opener burn/clicks are a junction PAIR — set both (the film-burn junction) or neither "
                "(a hard-cut concat)")
        return self


class SpecLogo(BaseModel):
    """The persistent corner logo over the BODY. `asset` is an inputs[].id of the logo PNG — the pod
    holds no brand profile and never reads one. `cover_hold` is the length of the cover end-card tail
    the logo must NOT cover (the end-card carries its own logo)."""

    model_config = ConfigDict(extra="forbid")

    asset: str = Field(min_length=1)
    corner: Literal["tl", "tr", "bl", "br"]
    width: int = Field(gt=0)
    opacity: float = Field(ge=0, le=1)
    margin: int = Field(ge=0)
    cover_hold: float = Field(ge=0)


class SpecWatermark(BaseModel):
    """The animated brand watermark. `sting`/`idle` are inputs[].id of the two alpha .webm clips; the
    chime is the sting's own audio track. `x`/`y` are NUMBERS in pixels, never ffmpeg expressions —
    an expression would be a filtergraph fragment crossing the seam."""

    model_config = ConfigDict(extra="forbid")

    sting: str = Field(min_length=1)
    idle: str = Field(min_length=1)
    width: int = Field(gt=0)
    margin: int = Field(ge=0)
    position: Literal["bottom-center", "bottom-right", "bottom-left",
                      "top-center", "top-right", "top-left", "center"]
    x: float | None = None
    y: float | None = None
    delay: float = Field(default=0.0, ge=0)
    chime: bool = True
    chime_volume: float = Field(default=1.0, gt=0, le=2)

    @model_validator(mode="after")
    def _xy_together(self) -> "SpecWatermark":
        if (self.x is None) != (self.y is None):
            raise ValueError("watermark x and y are one placement — pass both or neither")
        return self


class SpecLoudnorm(BaseModel):
    """Delivery loudness. `attenuate_only` is the PLANNER's verdict that the source clipped hot (it
    read the clipping sidecar the pod has never seen); the pod applies it, it does not re-decide it."""

    model_config = ConfigDict(extra="forbid")

    i: float
    tp: float
    lra: float = Field(gt=0)
    attenuate_only: bool = False


class SpecFinalize(BaseModel):
    """The delivery tail applied AFTER the composite: frame accents, body logo, animated watermark,
    delivery loudness. Every sub-block is independently optional, so `--no-logo` / `--no-watermark`
    are expressed by OMISSION rather than by a flag the pod has to interpret — the pod runs exactly
    the steps it was handed."""

    model_config = ConfigDict(extra="forbid")

    accents: list[SpecAccent] = Field(default_factory=list)
    logo: SpecLogo | None = None
    watermark: SpecWatermark | None = None
    loudnorm: SpecLoudnorm | None = None


class SpecPersonRing(BaseModel):
    """A contour around the person plane: the alpha dilated `width` px ((2w+1) square max — ffmpeg `dilation` x w),
    filled with `color` (#rrggbb); `glow` = Gaussian sigma px of a halo of the same colour (0 = none)."""

    model_config = ConfigDict(extra="forbid")

    color: str = Field(pattern=r"^#[0-9a-fA-F]{6}$")
    width: int = Field(ge=1, le=32)
    glow: float = Field(default=0.0, ge=0, le=40)


class SpecPersonShadow(BaseModel):
    """A drop shadow of the (ringed) person plane: offset dx/dy px, Gaussian sigma `blur` px, black at `opacity`."""

    model_config = ConfigDict(extra="forbid")

    dx: int = Field(ge=-64, le=64)
    dy: int = Field(ge=-64, le=64)
    blur: float = Field(ge=0, le=40)
    opacity: float = Field(gt=0, le=1)


class SpecPersonStyle(BaseModel):
    """super-director R6: how the person plane is drawn on `windows` (master clock, seconds, each inside
    person_matte.windows; b-roll cutaway windows are cut out like the plane's own): `gray` = the plane in black and white
    (U = V = 128), `frame_gray` = the composite under it too, `ring` / `shadow` = drawn under the plane."""

    model_config = ConfigDict(extra="forbid")

    windows: list[tuple[float, float]] = Field(min_length=1)
    gray: bool = False
    frame_gray: bool = False
    ring: SpecPersonRing | None = None
    shadow: SpecPersonShadow | None = None

    @model_validator(mode="after")
    def _windows_forward(self) -> "SpecPersonStyle":
        for a, b in self.windows:
            if a < 0 or b <= a:
                raise ValueError(f"person_matte style window [{a}, {b}] must be 0 <= start < end")
        return self


class SpecPersonMatte(BaseModel):
    """super-director v13 R4 person plane (contracts/spec.schema.json `person_matte`): `alpha` = the inputs[].id of
    the gray FFV1 alpha, frame-for-frame with timeline.segments[0].src; `windows` = [start, end) seconds on the master
    clock where the plane shows (the union of the matte layers' windows minus layouts / head-below)."""

    model_config = ConfigDict(extra="forbid")

    alpha: str = Field(min_length=1)
    windows: list[tuple[float, float]] = Field(min_length=1)
    # R6: the plane's looks per window (absent = the plain cut-out everywhere — the R4 graph, byte-identical)
    styles: list[SpecPersonStyle] | None = None

    @model_validator(mode="after")
    def _windows_forward(self) -> "SpecPersonMatte":
        for a, b in self.windows:
            if a < 0 or b <= a:
                raise ValueError(f"person_matte window [{a}, {b}] must be 0 <= start < end")
        return self


class Overlays(BaseModel):
    model_config = ConfigDict(extra="forbid")

    broll_final: SpecBrollFinal | None = None
    motion_plan: SpecMotionPlan | None = None
    music: SpecMusic | None = None
    cover: SpecCover | None = None
    trims: list[SpecTrim] | None = None
    sfx: list[SpecSfx] | None = None
    finalize: SpecFinalize | None = None
    opener: SpecOpener | None = None
    person_matte: SpecPersonMatte | None = None


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
    # cover = the standalone cover.png deliverable; `receipt` must stay OPTIONAL — an engine declaring
    # none renders exactly as before, which is the only thing that lets a new image serve an old engine.
    kind: Literal["proxy", "master", "cache", "cover", "presync", "receipt"]
    put_url: str = Field(min_length=1)


class RenderSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    spec_version: Literal[7]
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
            pm = self.overlays.person_matte
            if pm is not None:
                if pm.alpha not in ids:
                    raise ValueError(f"person_matte.alpha {pm.alpha!r} is not an inputs[].id")
                plates = {seg.src for seg in self.timeline.segments}
                if len(plates) != 1:
                    raise ValueError(f"person_matte pairs its alpha frame-for-frame with ONE plate; the timeline "
                                     f"cuts {sorted(plates)}")
            if self.overlays.broll_final is not None:
                for n, c in enumerate(self.overlays.broll_final.broll):
                    if c.clip not in ids:
                        raise ValueError(f"broll[{n}].clip {c.clip!r} is not an inputs[].id")
            for n, s in enumerate(self.overlays.sfx or []):
                if s.sound not in ids:
                    raise ValueError(f"sfx[{n}].sound {s.sound!r} is not an inputs[].id")
            fin = self.overlays.finalize
            if fin is not None:
                refs = [("logo.asset", fin.logo.asset)] if fin.logo is not None else []
                if fin.watermark is not None:
                    refs += [("watermark.sting", fin.watermark.sting),
                             ("watermark.idle", fin.watermark.idle)]
                for n, a in enumerate(fin.accents):
                    if a.kind == "film_burn":
                        # a.burn/a.clicks are `str | None`, but SpecAccent._film_burn_needs_both already
                        # refused a film_burn kind with either unset — narrow explicitly (symmetry with
                        # the engine mirror; this module is not in the mypy strict island itself, but the
                        # assert is free and keeps the two copies byte-identical).
                        assert a.burn is not None and a.clicks is not None
                        refs += [(f"accents[{n}].burn", a.burn), (f"accents[{n}].clicks", a.clicks)]
                for what, ref in refs:
                    if ref not in ids:
                        raise ValueError(f"finalize.{what} {ref!r} is not an inputs[].id")
            op = self.overlays.opener
            if op is not None:
                op_refs = [("opener.cold", op.cold)]
                if op.burn is not None:
                    op_refs.append(("opener.burn", op.burn))
                if op.clicks is not None:
                    op_refs.append(("opener.clicks", op.clicks))
                for what, ref in op_refs:
                    if ref not in ids:
                        raise ValueError(f"{what} {ref!r} is not an inputs[].id")
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
    # Emission COLUMNS to ship back; a sorted id SET is an alphabet, never text (no word/order/count crosses).
    keep_ids: list[int] | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def _windows_ok(self) -> "AlignParams":
        _spans_ok(self.windows, "windows")
        if self.keep_ids is not None:
            if any(b <= a for a, b in zip(self.keep_ids, self.keep_ids[1:])):
                raise ValueError("keep_ids must be sorted and unique (it is a column SET, not a sequence)")
            if self.keep_ids[0] < 0:
                raise ValueError("keep_ids must be non-negative vocab indices")
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


class ClipRankGroup(BaseModel):
    """One (intent, images) scoring unit. `intent` == "" means embed-only — the image tower is
    text-independent, so the group still yields embeddings; each score is null with an `empty_intent` error."""

    model_config = ConfigDict(extra="forbid")

    intent: str
    # Exactly one URL kind per group: presigned `image_urls` (fetched under the presigned-only rule) or
    # stock-provider `origin_urls` (the pod fetches the provider thumbnail itself; MISC-333).
    image_urls: list[str] | None = Field(default=None, min_length=1)
    origin_urls: list[str] | None = Field(default=None, min_length=1)
    image_cells: list[tuple[int, int, int, int, int, int] | None] | None = None

    @property
    def tile_urls(self) -> list[str]:
        """The group's tiles, whichever URL kind carries them."""
        return self.image_urls if self.image_urls is not None else self.origin_urls  # type: ignore[return-value]

    @model_validator(mode="after")
    def _cells_match(self) -> "ClipRankGroup":
        if (self.image_urls is None) == (self.origin_urls is None):
            raise ValueError("a clip_rank group carries exactly one of image_urls or origin_urls")
        if self.image_cells is not None and self.image_urls is None:
            raise ValueError("image_cells are allowed only with image_urls")
        if self.image_cells is not None and len(self.image_cells) != len(self.image_urls):
            raise ValueError("image_cells must be parallel to image_urls")
        for cell in self.image_cells or []:
            if cell is not None and (min(cell[:2]) < 0 or min(cell[2:]) <= 0):
                raise ValueError("image cell geometry must be non-negative with positive sizes")
        return self


class ClipRankJudgeTile(BaseModel):
    """One judge tile to make: the pod fetches the stock-provider ORIGIN url and PUTs one small JPEG of it to
    the presigned `put_url` (the judge keeps PER-TILE images, so it is one object per tile, never a sheet)."""

    model_config = ConfigDict(extra="forbid")

    origin_url: str = Field(min_length=1)
    put_url: str = Field(min_length=1)


class ClipRankParams(BaseModel):
    """Both SigLIP towers over MANY groups → cosines + image embeddings back. The reorder, the relevance
    floor and the MMR dedup are PLANNER decisions — no threshold is in this block by construction.

    A request is EITHER a rank (>=1 group, no judge_tiles) OR a judge-tiles request (`groups == []` and >=1
    judge tile). `groups` stays required so today's requests are unchanged."""

    model_config = ConfigDict(extra="forbid")

    groups: list[ClipRankGroup]
    judge_tiles: list[ClipRankJudgeTile] | None = Field(default=None, min_length=1)

    @model_validator(mode="before")
    @classmethod
    def _judge_tiles_absent_not_null(cls, data: Any) -> Any:
        # The schema has no type for an explicit null here (judge_tiles is an array when present, absent
        # otherwise) — an explicit null must fail the SAME way as the schema, not be read as "absent".
        if isinstance(data, dict) and data.get("judge_tiles", False) is None:
            raise ValueError("judge_tiles must be a non-empty array when present, never null — omit it instead")
        return data

    @model_validator(mode="after")
    def _groups_or_judge_tiles(self) -> "ClipRankParams":
        if self.judge_tiles is None and not self.groups:
            raise ValueError("a clip_rank request carries at least one group (or judge_tiles with groups == [])")
        if self.judge_tiles is not None and self.groups:
            raise ValueError("a judge_tiles request carries groups == [] — never groups and judge_tiles together")
        return self


class PersonMatteParams(BaseModel):
    """RVM person matte over ONE whole video (the plate a tier's layers composite over). Every number is the
    planner's: `downsample_ratio` is resolved from the frame size before the request is written. The pod PUTs
    the requested outputs to their own presigned URLs and the receipt (person_matte.schema.json) to put_url."""

    model_config = ConfigDict(extra="forbid")

    video_url: str = Field(min_length=1)
    downsample_ratio: float = Field(gt=0, le=1)
    # FINAL: the alpha as a lossless gray FFV1 Matroska, frame-for-frame with the input
    alpha_put_url: str | None = Field(default=None, min_length=1)
    # PREVIEW: the input's pixels under the alpha as a VP9-alpha (yuva420p) WebM, no audio
    cutout_put_url: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def _some_output(self) -> "PersonMatteParams":
        if self.alpha_put_url is None and self.cutout_put_url is None:
            raise ValueError("person_matte must request at least one output (alpha_put_url / cutout_put_url)")
        return self


class CleanPlateWindow(BaseModel):
    """One plate the pod writes: the layer window [t0, t1] (seconds on the video's own clock) and the presigned PNG
    target. `id` is the sd layer id — the engine reads the plate back by it."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
    t0: float = Field(ge=0)
    t1: float = Field(gt=0)
    put_url: str = Field(min_length=1)

    @model_validator(mode="after")
    def _forward(self) -> "CleanPlateWindow":
        if self.t1 <= self.t0:
            raise ValueError(f"clean_plate window {self.id!r} [{self.t0}, {self.t1}] must have t0 < t1")
        return self


class CleanPlateParams(BaseModel):
    """The clean background plate of ONE video, one plate per window (super-director v13 vertigo class): a per-pixel
    temporal median over the pixels the person matte does not cover, then LaMa for what is never uncovered. Every
    number is the planner's (registry/clean_plate.yaml); the matte is the one the person_matte kind already produced
    for the SAME video (`matte_kind` says how to read its alpha), so no segmentation runs here."""

    model_config = ConfigDict(extra="forbid")

    video_url: str = Field(min_length=1)
    matte_url: str = Field(min_length=1)
    matte_kind: Literal["cutout", "alpha"]
    windows: list[CleanPlateWindow] = Field(min_length=1, max_length=32)
    samples: int = Field(ge=3, le=48)              # frames sampled per window, evenly spread
    person_alpha: int = Field(ge=1, le=254)        # a pixel whose matte alpha is above this is the person
    dilate_px: int = Field(ge=0, le=64)            # the person's silhouette grows by this before it is excluded
    min_visible: int = Field(ge=1, le=48)          # a pixel seen as background in fewer samples is a hole
    stable_range: int = Field(ge=4, le=255)        # visible samples that disagree by more than this (0..255) are a hole
    inpaint_long_side: int = Field(ge=256, le=2048)  # LaMa works at most this big; the result is upsampled into the hole
    max_hole_fraction: float = Field(gt=0, le=1)   # more hole than this and the plate is refused, not hallucinated

    @model_validator(mode="after")
    def _consistent(self) -> "CleanPlateParams":
        if self.min_visible > self.samples:
            raise ValueError("clean_plate min_visible cannot exceed samples")
        ids = [w.id for w in self.windows]
        if len(set(ids)) != len(ids):
            raise ValueError("clean_plate window ids must be unique")
        return self


class WeightsRef(BaseModel):
    """Where the pod gets this job's checkpoint. A presigned GET for a tar of the model directory, plus
    that tar's sha256 — which is BOTH the integrity check and the pod's cache key, so a re-exported or
    revised checkpoint can never be served from a stale cache entry."""

    model_config = ConfigDict(extra="forbid")

    url: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size: int | None = Field(default=None, gt=0)


# face_probe's YuNet (227 KB), person_matte's RVM (7.9 MB fp16 TorchScript) and clean_plate's big-lama (206 MB
# TorchScript) stay baked in the image
_NEEDS_WEIGHTS = ("align", "clip_rank")


InferKind = Literal["align", "face_probe", "clip_rank", "person_matte", "clean_plate"]


class InferRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    infer_version: Literal[7]
    job_id: str = Field(min_length=1)
    kind: InferKind
    model: str = Field(min_length=1)
    put_url: str = Field(min_length=1)
    weights: WeightsRef | None = None
    align: AlignParams | None = None
    face_probe: FaceProbeParams | None = None
    clip_rank: ClipRankParams | None = None
    person_matte: PersonMatteParams | None = None
    clean_plate: CleanPlateParams | None = None

    @model_validator(mode="after")
    def _block_matches_kind(self) -> "InferRequest":
        blocks: dict[str, Any] = {"align": self.align, "face_probe": self.face_probe,
                                  "clip_rank": self.clip_rank, "person_matte": self.person_matte,
                                  "clean_plate": self.clean_plate}
        if blocks[self.kind] is None:
            raise ValueError(f"kind={self.kind} requires its params block")
        extra = [k for k, v in blocks.items() if k != self.kind and v is not None]
        if extra:
            raise ValueError(f"kind={self.kind} must not carry another kind's params block: {extra}")
        return self

    @model_validator(mode="after")
    def _weights_match_kind(self) -> "InferRequest":
        # The image carries no heavy checkpoint any more, so a missing weights block is not a default to
        # fall back on — it is an origin bug, and it must fail HERE rather than deep inside from_pretrained.
        if self.kind in _NEEDS_WEIGHTS and self.weights is None:
            raise ValueError(f"kind={self.kind} requires a weights block (nothing is baked in the image)")
        if self.kind not in _NEEDS_WEIGHTS and self.weights is not None:
            raise ValueError(f"kind={self.kind} uses a baked model and must not carry weights")
        return self


class InferTiming(BaseModel):
    model_config = ConfigDict(extra="forbid")

    infer_s: float = Field(ge=0)
    boot_s: float | None = Field(default=None, ge=0)


class InferResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    infer_version: Literal[7]
    job_id: str = Field(min_length=1)
    kind: InferKind
    status: Literal["ok", "error"]
    result_key: str | None = Field(default=None, min_length=1)
    # Echoed from the job corr_id for errors; a success also carries result_key as durable business identity.
    corr_id: str | None = Field(default=None, min_length=1)
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


class OpsPackRef(BaseModel):
    """Where the pod gets the OPERATION HANDLERS. Same class as WeightsRef and BundleRef, and
    deliberately the same three fields: a presigned GET for a CI-built tar, plus that tar's sha256, which
    is BOTH the integrity check and the cache key.

    This is what keeps `pod-agent` PUBLIC while the tuned implementations stay private. The public repo
    holds the executor skeleton — fetch, verify, dispatch, transport. Filtergraph construction, encoder
    profiles and camera-expression building ride in the pack. And because the pack arrives as a presigned
    URL, the pod needs NO registry credential to get it: the keyless invariant (CP_URL + JOB_TOKEN, and
    nothing else, ever) survives intact. A private image would have cost exactly one credential too many.
    """

    model_config = ConfigDict(extra="forbid")

    url: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size: int | None = Field(default=None, gt=0)


class OpBinding(BaseModel):
    """One end of an op's declared port, bound to something concrete.

    EXACTLY ONE of `url` / `from_step` / `path` is set, and which one it is IS the transport decision:

      url       — crosses R2. A presigned GET (input) or PUT (output). Costs a round trip.
      urls      — OUTPUT ONLY, and only for a port the registry declares `many`: N addresses for the N
                  files that port yields, in order. One step, N addressable results.
      from_step — the output of an earlier step in THIS chain, on THIS box. Costs a path lookup.
                  `from_port` picks WHICH of that step's outputs when the two ports are not named alike.
      path      — a working-set path already hydrated on the box.

    `retain`/`retained_on` are the THIRD transport answer, for an artifact whose whole audience is later
    chains on the SAME box. `from_step` cannot serve them (the runner rmtree's each chain's workspace) and a
    url makes the box pay a PUT plus a GET for bytes that never leave it. An output that declares
    `retain: "local"` is adopted into the pod's input cache under its `url`'s object key and NOT uploaded; a
    later chain that binds that url declares `retained_on: "<worker>"` and is refused, by name, if it woke up
    anywhere else. Both fields are optional and absent on every envelope that predates them, which is why
    `contracts/VERSION` does not move.

    `from_step` is the whole point. the engine's render-fleet budget doc measured a pipeline that runs
    ~9 min locally taking ~25 min split onto a pod, of which ~10 min was sequential per-stage R2 transport.
    Op granularity is FINER than stage granularity, so a design that gave every op its own round trip
    would multiply that regression rather than fix it. Here the unit that crosses R2 is the JOB; inside a
    job, adjacent ops hand off via shared local disk exactly as the engine's transport architecture doc requires
    ("when two adjacent stages share ONE box in ONE job they hand off via shared local disk — that is the
    correct place to cut R2, and only there").
    """

    model_config = ConfigDict(extra="forbid")

    port: str = Field(min_length=1)
    url: str | None = Field(default=None, min_length=1)
    # A ceiling on ONE step's upload burst, not on what any caller may ask for: the addresses are minted
    # before the step runs, so an unbounded list is an unbounded envelope.
    urls: list[str] | None = Field(default=None, min_length=1, max_length=256)
    from_step: str | None = Field(default=None, min_length=1)
    # WHICH output of the producer, when the two ports are not named alike — a producer with >1 declared
    # output is otherwise unreadable (the runner only guesses a name for a single-output step).
    from_port: str | None = Field(default=None, min_length=1)
    path: str | None = Field(default=None, min_length=1)
    sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    # OUTPUT only. "local" = adopt the produced file into THIS pod's input cache under `url`'s object key and
    # do not upload it. A closed vocabulary of one: the value must name a retention POLICY, so a future
    # "lease"/"session" scope is an enum member rather than a boolean that has to be reinterpreted.
    retain: Literal["local"] | None = Field(default=None)
    # INPUT only. The worker that retained this object, as the control plane names it. Present, it turns the
    # bind into a fence: the bytes are on ONE box and a chain that woke up elsewhere must be told so by name
    # rather than discover a 404 (or, worse, silently re-render).
    retained_on: str | None = Field(default=None, min_length=1, pattern=r"^[A-Za-z0-9_.:-]+$")

    @model_validator(mode="after")
    def _exactly_one_source(self) -> "OpBinding":
        set_ = [n for n, v in (("url", self.url), ("urls", self.urls), ("from_step", self.from_step),
                               ("path", self.path)) if v is not None]
        if len(set_) != 1:
            raise ValueError(
                f"binding {self.port!r} must set exactly one of url/urls/from_step/path, got {set_}")
        if self.urls is not None and any(not u for u in self.urls):
            raise ValueError(f"binding {self.port!r}: an empty address in urls addresses nothing")
        if self.from_port is not None and self.from_step is None:
            raise ValueError(f"binding {self.port!r} names from_port with no from_step to read it from")
        # Both retention fields hang off the OBJECT KEY, which only a single `url` carries: `urls` is N keys
        # with one flag between them, and from_step/path name no key at all.
        if self.retain is not None and self.url is None:
            raise ValueError(f"binding {self.port!r} declares retain={self.retain!r} but binds no `url` — "
                             f"retention keys on the object address the file WOULD have had")
        if self.retained_on is not None and self.url is None:
            raise ValueError(f"binding {self.port!r} names retained_on with no `url` to identify the "
                             f"retained object")
        if self.retain is not None and self.retained_on is not None:
            raise ValueError(f"binding {self.port!r} both retains and reads a retained object; one binding "
                             f"is one end of one port")
        return self


class OpStep(BaseModel):
    """One op invocation inside a chain."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    op: str = Field(min_length=1, pattern=r"^[a-z][a-z0-9_]*\.[a-z][a-z0-9_]*$")
    needs: list[str] = Field(default_factory=list)
    # NOT a per-op Pydantic model. `params` is validated against the registry's schema fragment
    # (podagent.ops.registry.validate_params) so a NEW OP ADDS NO MODEL HERE — the envelope stays closed,
    # the op's own surface is declared once in contracts/ops/<op>.json and read by both sides.
    params: dict[str, Any] = Field(default_factory=dict)
    inputs: list[OpBinding] = Field(default_factory=list)
    outputs: list[OpBinding] = Field(default_factory=list)
    # A FAN-OUT chain does the same independent work N times — one preview per candidate — and the failure
    # of one arm says nothing about the other N-1. Marked optional, a step's failure costs that step and the
    # steps that read it, and the chain runs on; unmarked (the default), any failure is the chain's, which is
    # what a render pipeline needs. The caller decides which arms actually delivered by looking at outputs.
    optional: bool = False

    @model_validator(mode="after")
    def _ports_unique(self) -> "OpStep":
        for side, binds in (("inputs", self.inputs), ("outputs", self.outputs)):
            names = [b.port for b in binds]
            if len(set(names)) != len(names):
                raise ValueError(f"step {self.id!r}: duplicate {side} port")
        for b in self.outputs:
            if b.from_step is not None or b.from_port is not None:
                raise ValueError(f"step {self.id!r}: an output cannot bind from_step/from_port")
            if b.retained_on is not None:
                raise ValueError(f"step {self.id!r}: output {b.port!r} names retained_on — an output is "
                                 f"where retention is DECLARED (`retain`), never where it is read")
        for b in self.inputs:
            if b.retain is not None:
                raise ValueError(f"step {self.id!r}: input {b.port!r} declares retain — only an output the "
                                 f"handler actually produces can be kept on the box")
            # An input is ONE file: a port fed from N addresses has no defined order to be read in, and the
            # runner would have to invent one. N inputs are N ports.
            if b.urls is not None:
                raise ValueError(f"step {self.id!r}: input {b.port!r} binds urls — only an output may")
        return self


class OpChain(BaseModel):
    """A DAG of ops executed on ONE box, in ONE job, with local-disk handoff between steps.

    Independent steps run CONCURRENTLY (the runner walks `needs`), so the chain neither serialises what
    the DAG says is parallel nor pays transport for what stays on the box.
    """

    model_config = ConfigDict(extra="forbid")

    chain_version: Literal[1] = 1
    job_id: str = Field(min_length=1)
    pack: OpsPackRef
    steps: list[OpStep] = Field(min_length=1)

    @model_validator(mode="after")
    def _dag_is_sound(self) -> "OpChain":
        ids = [s.id for s in self.steps]
        if len(set(ids)) != len(ids):
            raise ValueError("steps[].id must be unique")
        known: set[str] = set()
        seen = set(ids)
        for s in self.steps:
            for n in s.needs:
                if n not in seen:
                    raise ValueError(f"step {s.id!r} needs unknown step {n!r}")
            for b in s.inputs:
                if b.from_step is not None and b.from_step not in seen:
                    raise ValueError(f"step {s.id!r} binds unknown step {b.from_step!r}")
        # topological soundness: a cycle would deadlock the runner, so refuse it at validation where the
        # error names the steps, not at run time where it looks like a hang on a rented box.
        pending = {s.id: set(s.needs) | {b.from_step for b in s.inputs if b.from_step} for s in self.steps}
        while pending:
            ready = {sid for sid, deps in pending.items() if not (deps - known)}
            if not ready:
                raise ValueError(f"steps form a cycle: {sorted(pending)}")
            known |= ready
            for sid in ready:
                pending.pop(sid)
        # A step reading another step's output MUST also depend on it, or the runner may schedule the two
        # concurrently and the reader races a file that is still being written.
        for s in self.steps:
            for b in s.inputs:
                if b.from_step is not None and b.from_step not in s.needs:
                    raise ValueError(
                        f"step {s.id!r} reads {b.from_step!r} but does not list it in needs — that is a "
                        f"race, not an optimisation")
        return self


class PodJob(BaseModel):
    """The control-plane→pod job envelope (typed EventStream job frame). No version const of its
    own — request/spec/chain each pin their own version; contracts/VERSION is the shared pin."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["infer", "render", "ops"]
    request: InferRequest | None = None
    spec: RenderSpec | None = None
    chain: OpChain | None = None
    # Pool routing. Every terminal must have a stable dedupe/demux identity; there is no positional fallback.
    session_id: str = Field(min_length=1)
    corr_id: str = Field(min_length=1)
    # Optional physical-worker fence used only for infrastructure warm-up.
    # The API routes a targeted envelope to the named worker; the pod keeps
    # the field in its strict model for contract parity and dispatches the
    # admitted product payload exactly as usual.
    target_worker_id: str | None = Field(
        default=None, min_length=1, pattern=r"^[A-Za-z0-9_.:-]+$"
    )

    @model_validator(mode="after")
    def _block_matches_type(self) -> "PodJob":
        blocks = {"infer": self.request, "render": self.spec, "ops": self.chain}
        want = blocks.pop(self.type)
        if want is None:
            raise ValueError(f"type={self.type} requires its matching block")
        extra = [n for n, v in blocks.items() if v is not None]
        if extra:
            raise ValueError(f"type={self.type} must not carry the other type's block ({extra})")
        return self


class FaceProbePayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str = Field(min_length=1)
    width: int = Field(ge=2)
    height: int = Field(ge=2)
    shots: list[ProbeShot] = Field(min_length=1)


class MatteOutput(BaseModel):
    """One PUT person_matte output, as the pod wrote it (the engine verifies the object against these)."""

    model_config = ConfigDict(extra="forbid")

    bytes: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class PersonMatteTimings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_s: float = Field(ge=0)          # GPU forward + device copies, summed over frames
    first_frame_s: float = Field(ge=0)    # the cudnn autotune frame, inside model_s
    encode_drain_s: float = Field(ge=0)   # encoders finishing after the last frame
    wall_s: float = Field(ge=0)           # download excluded, upload excluded


class PersonMattePayload(BaseModel):
    """person_matte.schema.json — the receipt the pod PUTs to put_url. `frames` is what the decoder delivered
    AND what each output carries (the pod refuses a mismatch, so a short matte can never be composited)."""

    model_config = ConfigDict(extra="forbid")

    model: str = Field(min_length=1)
    model_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    width: int = Field(ge=2)
    height: int = Field(ge=2)
    fps: str = Field(pattern=r"^[1-9][0-9]*(/[1-9][0-9]*)?$")   # the input's r_frame_rate, verbatim
    frames: int = Field(ge=1)
    downsample_ratio: float = Field(gt=0, le=1)
    alpha: MatteOutput | None = None
    cutout: MatteOutput | None = None
    timings: PersonMatteTimings

    @model_validator(mode="after")
    def _some_output(self) -> "PersonMattePayload":
        if self.alpha is None and self.cutout is None:
            raise ValueError("a person_matte receipt names at least one output")
        return self


class CleanPlateTimings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decode_s: float = Field(ge=0)     # ffmpeg sampling of the plate video and the matte, all windows
    median_s: float = Field(ge=0)     # the masked temporal median on the card
    inpaint_s: float = Field(ge=0)    # LaMa, all windows (0 when no window had a hole)
    wall_s: float = Field(ge=0)       # download excluded, upload excluded


class CleanPlateOut(BaseModel):
    """One PUT plate, as the pod wrote it. `covered` + `inpainted` = 1: the share of the frame taken from the median
    of real background pixels, and the share LaMa drew."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    bytes: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    samples: int = Field(ge=1)
    covered: float = Field(ge=0, le=1)
    inpainted: float = Field(ge=0, le=1)


class CleanPlateRefusal(BaseModel):
    """A window the pod would not draw a plate for, and why (the other windows are unaffected)."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    reason: str = Field(min_length=1)


class CleanPlatePayload(BaseModel):
    """clean_plate.schema.json — the receipt the pod PUTs to put_url. Every requested window is in `plates` (drawn and
    PUT) or in `refused` (named, with the reason) — never in neither, never in both."""

    model_config = ConfigDict(extra="forbid")

    model: str = Field(min_length=1)
    model_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    width: int = Field(ge=2)
    height: int = Field(ge=2)
    plates: list[CleanPlateOut] = Field(default_factory=list)
    refused: list[CleanPlateRefusal] = Field(default_factory=list)
    timings: CleanPlateTimings

    @model_validator(mode="after")
    def _every_window_once(self) -> "CleanPlatePayload":
        ids = [p.id for p in self.plates] + [r.id for r in self.refused]
        if not ids:
            raise ValueError("a clean_plate receipt names at least one window")
        if len(set(ids)) != len(ids):
            raise ValueError("a clean_plate window is drawn or refused, once")
        return self


def _json_number(value: Any) -> Any:
    # JSON Schema's number/integer never match a boolean or a string; pydantic's lax mode would coerce both
    # (True -> 1, "5" -> 5). Refuse them here so the mirror accepts exactly what the schema accepts.
    if isinstance(value, (bool, str, bytes, bytearray)):
        raise ValueError(f"{value!r} is not a JSON number")
    return value


# A JSON integer / number exactly as the schema types read it: 5 and 5.0 are integers, True and "5" are not.
JsonIntT = Annotated[int, BeforeValidator(_json_number)]
JsonNumberT = Annotated[float, BeforeValidator(_json_number)]


ClipRankTileErrorKind = Literal["url_not_allowed", "fetch_timeout", "fetch_failed", "decode_error",
                                "empty_intent", "non_finite", "origin_fetch_unavailable", "origin_refused",
                                "put_failed"]


class ClipRankTileError(BaseModel):
    """Why one tile has no score (and, unless kind=empty_intent, no embed). A failure is NAMED, never a
    valid-looking number: -1.0 once read as "irrelevant" for 11 days while every tile was UrlNotAllowed."""

    model_config = ConfigDict(extra="forbid")

    kind: ClipRankTileErrorKind
    detail: str = Field(min_length=1)


class ClipRankTileMeasure(BaseModel):
    """One tile's measured size as the pod fetched it: encoded bytes and decoded pixel dimensions."""

    model_config = ConfigDict(extra="forbid")

    bytes: int = Field(ge=0)
    w: int = Field(gt=0)
    h: int = Field(gt=0)


class ClipRankGroupResult(BaseModel):
    """One group's raw SigLIP output; order mirrors the request group's tiles (image_urls or origin_urls).

    `scores[j]` is null exactly when `errors[j]` names why. A group in which NO tile produced anything (no
    score and no embed) is status=error carrying its first tile's reason — an all-failed group is an error,
    never a row of misses."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["ok", "error"]
    error: ClipRankTileError | None = None
    scores: list[float | None]
    embeds: list[list[float] | None]
    errors: list[ClipRankTileError | None]
    # Optional, parallel to the tiles when present: null for a tile the pod did not measure.
    measures: list[ClipRankTileMeasure | None] | None = None

    @model_validator(mode="after")
    def _failures_are_named(self) -> "ClipRankGroupResult":
        if not len(self.scores) == len(self.embeds) == len(self.errors):
            raise ValueError("scores, embeds and errors must be parallel to the request group's tiles")
        if self.measures is not None and len(self.measures) != len(self.scores):
            raise ValueError("measures must be parallel to the request group's tiles")
        for j, (score, embed, err) in enumerate(zip(self.scores, self.embeds, self.errors)):
            if (score is None) != (err is not None):
                raise ValueError(f"tile {j}: score is null exactly when errors[{j}] names why")
            # NaN/inf serialize to JSON null — an UNNAMED failure; refuse them so the writer must name them.
            if score is not None and not math.isfinite(score):
                raise ValueError(f"tile {j}: score {score} is not finite — name it in errors[{j}] instead")
            if embed is not None and not all(math.isfinite(x) for x in embed):
                raise ValueError(f"tile {j}: embedding is not finite — name it in errors[{j}] instead")
            if embed is None and (err is None or err.kind == "empty_intent"):
                raise ValueError(f"tile {j}: a null embed needs errors[{j}] naming a fetch/decode/forward failure")
        dead = bool(self.scores) and all(s is None and e is None for s, e in zip(self.scores, self.embeds))
        if self.status == "error":
            if self.error is None:
                raise ValueError("status=error requires error")
            if not dead:
                raise ValueError("status=error is only for a group in which no tile produced anything")
        else:
            if self.error is not None:
                raise ValueError("status=ok must not carry error")
            if dead:
                raise ValueError("a group in which no tile produced anything is status=error, not ok")
        return self


class ClipRankJudgeTileResult(BaseModel):
    """One judge tile's outcome, in request order: ok carries the PUT JPEG's bytes and size and whether the
    origin fetch hit the pod's cache; error names why no tile was written. A failure is never a size."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["ok", "error"]
    error: ClipRankTileError | None = None
    bytes: JsonIntT | None = Field(default=None, ge=0)
    w: JsonIntT | None = Field(default=None, gt=0)
    h: JsonIntT | None = Field(default=None, gt=0)
    cache: Literal["hit", "miss"] | None = None

    @model_validator(mode="before")
    @classmethod
    def _absent_not_null(cls, data: Any) -> Any:
        # Rule shared with the schema: a judge-tile result never carries an explicit null — a field is
        # present with its value or absent (an ok tile has no `error` key at all, `"error": null` included).
        if isinstance(data, dict):
            nulls = sorted(k for k, v in data.items() if v is None)
            if nulls:
                raise ValueError(f"a judge-tile result omits a field instead of null: {nulls}")
        return data

    @model_validator(mode="after")
    def _ok_is_measured(self) -> "ClipRankJudgeTileResult":
        measured = (self.bytes, self.w, self.h, self.cache)
        if self.status == "ok":
            if self.error is not None:
                raise ValueError("an ok judge tile must not carry error")
            if any(v is None for v in measured):
                raise ValueError("an ok judge tile carries bytes, w, h and cache")
        else:
            if self.error is None:
                raise ValueError("an error judge tile names its error")
            if any(v is not None for v in measured):
                raise ValueError("an error judge tile carries no bytes, w, h or cache")
        return self


class ClipRankPayload(BaseModel):
    """The JSON the pod PUTs for kind=clip_rank — numbers only, no ranking and no threshold.

    EITHER rank results (>=1 group, no judge_tiles) OR judge-tile results (`groups == []`, >=1 judge tile).
    `timings` are the pod's own request walls and counts known before this payload's PUT."""

    model_config = ConfigDict(extra="forbid")

    model: str = Field(min_length=1)
    groups: list[ClipRankGroupResult]
    judge_tiles: list[ClipRankJudgeTileResult] | None = Field(default=None, min_length=1)
    timings: dict[str, JsonNumberT] | None = None

    @model_validator(mode="before")
    @classmethod
    def _judge_tiles_absent_not_null(cls, data: Any) -> Any:
        # Same mirror rule as ClipRankParams: the schema has no null for judge_tiles or timings, only
        # value-or-absent.
        if isinstance(data, dict):
            for key in ("judge_tiles", "timings"):
                if data.get(key, False) is None:
                    raise ValueError(f"{key} must be present with a value or absent, never null — omit it")
        return data

    @model_validator(mode="after")
    def _groups_or_judge_tiles(self) -> "ClipRankPayload":
        if self.judge_tiles is None and not self.groups:
            raise ValueError("a clip_rank payload carries at least one group (or judge_tiles with groups == [])")
        if self.judge_tiles is not None and self.groups:
            raise ValueError("a judge_tiles payload carries groups == [] — never groups and judge_tiles together")
        for name, value in (self.timings or {}).items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"timings[{name!r}] = {value} must be finite and >= 0")
        return self

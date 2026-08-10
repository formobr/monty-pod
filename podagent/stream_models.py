"""Strict typed payloads for the single pod EventStream seam."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, RootModel, StringConstraints, model_validator
from typing_extensions import Annotated

from .models import PodJob


StreamID = Annotated[
    str,
    StringConstraints(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.-]+$"),
]
AttemptID = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{32}$")]
DeliveryID = Annotated[str, StringConstraints(min_length=1)]


class StreamEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stage: str = Field(min_length=1)
    status: Literal["done", "ok", "step", "error"]
    job_id: str | None = Field(default=None, min_length=1)
    session_id: str | None = Field(default=None, min_length=1)
    corr_id: str | None = Field(default=None, min_length=1)
    step: str | None = Field(default=None, min_length=1)
    op: str | None = Field(default=None, min_length=1)
    phase: str | None = Field(default=None, min_length=1)
    outcome: Literal["ok", "error"] | None = None
    optional: bool | None = None
    steps: list[str] | None = None
    skipped: list[str] | None = None
    outputs: Any | None = None
    error: str | None = None
    error_type: str | None = Field(default=None, min_length=1)
    timings: dict[str, Any] | None = None
    timeline: dict[str, Any] | None = None
    capacity: dict[str, Any] | None = None
    ts: datetime | None = None

    @model_validator(mode="before")
    @classmethod
    def _known_fields_are_not_null(cls, value: Any) -> Any:
        if isinstance(value, dict):
            nulls = sorted(k for k, v in value.items() if v is None and k != "outputs")
            if nulls:
                raise ValueError(f"event fields may be omitted but not null: {nulls}")
        return value


class StreamResult(BaseModel):
    # Result bodies remain extensible domain payloads. The transport address and discriminator are closed.
    model_config = ConfigDict(extra="allow")

    job_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    corr_id: str = Field(min_length=1)
    status: Literal["ok", "error"]
    kind: str | None = Field(default=None, min_length=1)
    stage: str | None = Field(default=None, min_length=1)
    result_key: str | None = Field(default=None, min_length=1)
    timing: dict[str, Any] | None = None
    timings: dict[str, Any] | None = None
    error: str | None = None
    timeline: dict[str, Any] | None = None

    @model_validator(mode="before")
    @classmethod
    def _known_fields_are_not_null(cls, value: Any) -> Any:
        known = {"kind", "stage", "result_key", "timing", "timings", "error", "timeline"}
        if isinstance(value, dict):
            nulls = sorted(k for k in known if k in value and value[k] is None)
            if nulls:
                raise ValueError(f"result fields may be omitted but not null: {nulls}")
        return value

    @model_validator(mode="after")
    def _wire_shape(self) -> "StreamResult":
        if (self.kind is None) == (self.stage is None):
            raise ValueError("result requires exactly one of kind or stage")
        if self.timing is not None and self.timings is not None:
            raise ValueError("result may carry timing or timings, not both")
        if self.result_key is not None and self.result_key != self.corr_id:
            raise ValueError("result_key, when present, must equal corr_id")
        return self


class StreamAck(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["ack"]
    stream_id: StreamID
    seq: int = Field(ge=1)
    status: int = Field(ge=100, le=599)
    server_recv_unix_ns: int = Field(ge=0)
    server_send_unix_ns: int = Field(ge=0)
    duplicate: bool | None = None
    error: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _optional_fields_are_not_null(cls, value: Any) -> Any:
        if isinstance(value, dict):
            nulls = sorted(k for k in ("duplicate", "error") if k in value and value[k] is None)
            if nulls:
                raise ValueError(f"ACK fields may be omitted but not null: {nulls}")
        return value

    @model_validator(mode="after")
    def _clock_order(self) -> "StreamAck":
        if self.server_send_unix_ns < self.server_recv_unix_ns:
            raise ValueError("ACK server_send_unix_ns precedes server_recv_unix_ns")
        return self


class JobTimeline(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enqueue_min_unix_ns: int = Field(ge=0)
    enqueue_max_unix_ns: int = Field(ge=0)
    claim_min_unix_ns: int = Field(ge=0)
    claim_max_unix_ns: int = Field(ge=0)
    socket_write_min_unix_ns: int = Field(ge=0)

    @model_validator(mode="after")
    def _bounds(self) -> "JobTimeline":
        if self.enqueue_max_unix_ns < self.enqueue_min_unix_ns:
            raise ValueError("enqueue bounds are reversed")
        if self.claim_max_unix_ns < self.claim_min_unix_ns:
            raise ValueError("claim bounds are reversed")
        return self


class StreamJob(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["job"]
    delivery_id: DeliveryID
    attempt_id: AttemptID
    stream_id: StreamID
    seq: int = Field(ge=1)
    replayed: bool
    timeline: JobTimeline
    job: PodJob

    @model_validator(mode="before")
    @classmethod
    def _job_blocks_are_not_null(cls, value: Any) -> Any:
        if isinstance(value, dict) and isinstance(value.get("job"), dict):
            nulls = sorted(
                k for k in ("request", "spec", "chain")
                if k in value["job"] and value["job"][k] is None)
            if nulls:
                raise ValueError(f"job blocks may be omitted but not null: {nulls}")
        return value

    @model_validator(mode="after")
    def _one_identity(self) -> "StreamJob":
        if self.delivery_id != self.job.corr_id:
            raise ValueError("job delivery_id must equal job.corr_id")
        return self


class StreamEventFrame(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["event"]
    stream_id: StreamID
    seq: int = Field(ge=1)
    client_send_mono_ns: int = Field(ge=0)
    event: StreamEvent


class StreamResultFrame(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["result"]
    stream_id: StreamID
    seq: int = Field(ge=1)
    client_send_mono_ns: int = Field(ge=0)
    attempt_id: AttemptID
    result: StreamResult


class JobAckPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    delivery_id: DeliveryID
    corr_id: str = Field(min_length=1)
    attempt_id: AttemptID
    client_recv_mono_ns: int = Field(ge=0)

    @model_validator(mode="after")
    def _one_identity(self) -> "JobAckPayload":
        if self.delivery_id != self.corr_id:
            raise ValueError("job_ack delivery_id must equal corr_id")
        return self


class StreamJobAckFrame(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["job_ack"]
    stream_id: StreamID
    seq: int = Field(ge=1)
    client_send_mono_ns: int = Field(ge=0)
    job_ack: JobAckPayload


class PodStreamFrame(RootModel[StreamEventFrame | StreamResultFrame | StreamJobAckFrame]):
    pass


class PodStreamServerFrame(RootModel[StreamAck | StreamJob]):
    pass


def event_payload(value: Any) -> dict[str, Any]:
    return StreamEvent.model_validate(value).model_dump(exclude_none=True, mode="json")


def result_payload(value: Any) -> dict[str, Any]:
    return StreamResult.model_validate(value).model_dump(exclude_none=True, mode="json")


def job_ack_payload(value: Any) -> dict[str, Any]:
    return JobAckPayload.model_validate(value).model_dump(mode="json")

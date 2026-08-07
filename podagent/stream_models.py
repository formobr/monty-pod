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

    @model_validator(mode="before")
    @classmethod
    def _known_fields_are_not_null(cls, value: Any) -> Any:
        known = {"kind", "stage", "result_key", "timing", "timings", "error"}
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


class StreamJob(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["job"]
    delivery_id: DeliveryID
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
    event: StreamEvent


class StreamResultFrame(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["result"]
    stream_id: StreamID
    seq: int = Field(ge=1)
    result: StreamResult


class PodStreamFrame(RootModel[StreamEventFrame | StreamResultFrame]):
    pass


class PodStreamServerFrame(RootModel[StreamAck | StreamJob]):
    pass


def event_payload(value: Any) -> dict[str, Any]:
    return StreamEvent.model_validate(value).model_dump(exclude_none=True, mode="json")


def result_payload(value: Any) -> dict[str, Any]:
    return StreamResult.model_validate(value).model_dump(exclude_none=True, mode="json")

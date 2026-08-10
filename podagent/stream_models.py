"""Semantic facade over generated structural models for the shared pod EventStream."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, RootModel, model_validator

from .models import PodJob
from .wire_generated import (
    JobAckPayloadFields,
    JobTimelineFields,
    StreamAckFields,
    StreamEventFields,
    StreamEventFrameFields,
    StreamJobAckFrameFields,
    StreamJobFields,
    StreamResultFields,
    StreamResultFrameFields,
)


class StreamEvent(StreamEventFields, BaseModel):
    pass


class StreamResult(StreamResultFields, BaseModel):
    @model_validator(mode="after")
    def _wire_shape(self) -> "StreamResult":
        if (self.kind is None) == (self.stage is None):
            raise ValueError("result requires exactly one of kind or stage")
        if self.timing is not None and self.timings is not None:
            raise ValueError("result may carry timing or timings, not both")
        if self.result_key is not None and self.result_key != self.corr_id:
            raise ValueError("result_key, when present, must equal corr_id")
        return self


class StreamAck(StreamAckFields, BaseModel):
    @model_validator(mode="after")
    def _clock_order(self) -> "StreamAck":
        if self.server_send_unix_ns < self.server_recv_unix_ns:
            raise ValueError("ACK server_send_unix_ns precedes server_recv_unix_ns")
        return self


class JobTimeline(JobTimelineFields, BaseModel):
    @model_validator(mode="after")
    def _bounds(self) -> "JobTimeline":
        if self.enqueue_max_unix_ns < self.enqueue_min_unix_ns:
            raise ValueError("enqueue bounds are reversed")
        if self.claim_max_unix_ns < self.claim_min_unix_ns:
            raise ValueError("claim bounds are reversed")
        return self


class StreamJob(StreamJobFields, BaseModel):
    @model_validator(mode="before")
    @classmethod
    def _job_blocks_are_not_null(cls, value: Any) -> Any:
        if isinstance(value, dict) and isinstance(value.get("job"), dict):
            nulls = sorted(
                key for key in ("request", "spec", "chain")
                if key in value["job"] and value["job"][key] is None
            )
            if nulls:
                raise ValueError(f"job blocks may be omitted but not null: {nulls}")
        return value

    @model_validator(mode="after")
    def _one_identity(self) -> "StreamJob":
        if self.delivery_id != self.job.corr_id:
            raise ValueError("job delivery_id must equal job.corr_id")
        return self


class StreamEventFrame(StreamEventFrameFields, BaseModel):
    pass


class StreamResultFrame(StreamResultFrameFields, BaseModel):
    pass


class JobAckPayload(JobAckPayloadFields, BaseModel):
    @model_validator(mode="after")
    def _one_identity(self) -> "JobAckPayload":
        if self.delivery_id != self.corr_id:
            raise ValueError("job_ack delivery_id must equal corr_id")
        return self


class StreamJobAckFrame(StreamJobAckFrameFields, BaseModel):
    pass


class PodStreamFrame(RootModel[StreamEventFrame | StreamResultFrame | StreamJobAckFrame]):
    pass


class PodStreamServerFrame(RootModel[StreamAck | StreamJob]):
    pass


for _model in (StreamEventFrame, StreamResultFrame, StreamJobAckFrame, StreamJob):
    _model.model_rebuild(_types_namespace=globals())


def event_payload(value: Any) -> dict[str, Any]:
    return StreamEvent.model_validate(value).model_dump(exclude_none=True, mode="json")


def result_payload(value: Any) -> dict[str, Any]:
    return StreamResult.model_validate(value).model_dump(exclude_none=True, mode="json")


def job_ack_payload(value: Any) -> dict[str, Any]:
    return JobAckPayload.model_validate(value).model_dump(mode="json")

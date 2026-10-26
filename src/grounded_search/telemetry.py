"""OpenTelemetry setup: Strands emits spans for every invocation, model call and tool call.

Exporters are configured via env vars (see config.py):
- OTEL_TRACES_FILE (default traces.jsonl) appends one JSON span per line to a file. On by default,
  needs no backend, and keeps the console readable.
- OTEL_CONSOLE_EXPORT=true also prints spans to stdout (very verbose).
- LANGFUSE_PUBLIC_KEY + LANGFUSE_SECRET_KEY send them to Langfuse's OTLP endpoint.
- OTEL_EXPORTER_OTLP_ENDPOINT (+ OTEL_EXPORTER_OTLP_HEADERS) sends them to any other OTLP
  backend (Arize Phoenix, Jaeger, an OTel collector, ...).
"""
from __future__ import annotations

import base64
import logging

from opentelemetry.sdk.trace import ReadableSpan
from strands.telemetry import StrandsTelemetry

from .config import settings

logger = logging.getLogger(__name__)

_telemetry: StrandsTelemetry | None = None
_active: list[str] = []
_traces_file = None


def _one_line(span: ReadableSpan) -> str:
    return span.to_json(indent=None) + "\n"


def _langfuse_exporter_kwargs() -> dict:
    credentials = f"{settings.langfuse_public_key}:{settings.langfuse_secret_key}".encode()
    return {
        "endpoint": f"{settings.langfuse_host.rstrip('/')}/api/public/otel/v1/traces",
        "headers": {"Authorization": f"Basic {base64.b64encode(credentials).decode()}"},
    }


def setup_telemetry() -> list[str]:
    """Configures the exporters enabled in settings. Returns a description of each active one."""
    global _telemetry, _active, _traces_file
    if _telemetry is not None:
        return _active

    use_langfuse = bool(settings.langfuse_public_key and settings.langfuse_secret_key)
    use_otlp = bool(settings.otel_exporter_otlp_endpoint)
    if not (settings.otel_traces_file or settings.otel_console_export or use_langfuse or use_otlp):
        return []

    telemetry = StrandsTelemetry()
    active: list[str] = []
    if settings.otel_traces_file:
        _traces_file = open(settings.otel_traces_file, "a", encoding="utf-8")
        telemetry.setup_console_exporter(out=_traces_file, formatter=_one_line)
        active.append(f"file {settings.otel_traces_file}")
    if settings.otel_console_export:
        telemetry.setup_console_exporter()
        active.append("stdout")
    try:
        if use_langfuse:
            telemetry.setup_otlp_exporter(**_langfuse_exporter_kwargs())
            active.append(f"Langfuse {settings.langfuse_host}")
        elif use_otlp:
            # The OTLP exporter reads OTEL_EXPORTER_OTLP_ENDPOINT / _HEADERS from the environment.
            telemetry.setup_otlp_exporter()
            active.append(f"OTLP {settings.otel_exporter_otlp_endpoint}")
    except ImportError:
        logger.warning("OTLP exporter not installed: pip install -e '.[otel]'")

    _telemetry, _active = telemetry, active
    return active


def flush_telemetry() -> None:
    """Exports pending spans; the OTLP exporter batches them, so call this before exiting."""
    if _telemetry is not None:
        _telemetry.tracer_provider.force_flush()

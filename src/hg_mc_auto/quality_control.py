from __future__ import annotations

import json
import os
import csv
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping


SCHEMA_VERSION = "qc.v1"
ALLOWED_SEVERITIES = {"info", "warning", "critical"}


@dataclass(frozen=True)
class QCFlag:
    code: str
    severity: str
    message: str = ""
    source: str = "rule"
    evidence: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.severity not in ALLOWED_SEVERITIES:
            raise ValueError(f"Unsupported QC flag severity: {self.severity}")
        if not self.code.strip():
            raise ValueError("QC flag code cannot be empty")


@dataclass
class QCPacket:
    element: str
    run_id: str
    metrics: Dict[str, Any]
    results: Dict[str, Any]
    flags: List[QCFlag]
    schema_version: str = SCHEMA_VERSION
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    human_confirmation: Dict[str, Any] = field(
        default_factory=lambda: {
            "required": True,
            "status": "pending",
            "approved_by": None,
            "approved_at": None,
        }
    )

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["flags"] = [asdict(flag) for flag in self.flags]
        return payload

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2, sort_keys=True)


def normalize_qc_result(raw_flag: Mapping[str, Any]) -> QCFlag:
    """Convert a detector result into the stable QC flag contract."""
    return QCFlag(
        code=str(raw_flag.get("code", "UNKNOWN")),
        severity=str(raw_flag.get("severity", "warning")).lower(),
        message=str(raw_flag.get("message", "")),
        source=str(raw_flag.get("source", "rule")),
        evidence=dict(raw_flag.get("evidence", {})),
    )


def build_qc_packet(
    *,
    element: str,
    run_id: str,
    metrics: Mapping[str, Any],
    results: Mapping[str, Any],
    flags: List[Mapping[str, Any] | QCFlag],
) -> QCPacket:
    normalized_flags = [
        flag if isinstance(flag, QCFlag) else normalize_qc_result(flag)
        for flag in flags
    ]
    return QCPacket(
        element=element.lower(),
        run_id=run_id,
        metrics=dict(metrics),
        results=dict(results),
        flags=normalized_flags,
    )


def _read_result_rows(path: Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            return [payload]
        if isinstance(payload, list) and all(isinstance(row, dict) for row in payload):
            return payload
        raise ValueError("JSON result file must contain an object or a list of objects")
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8-sig") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    if path.suffix.lower() in {".xlsx", ".xls"}:
        try:
            import pandas as pd
        except ImportError as exc:
            raise ValueError("Excel result files require pandas") from exc
        return pd.read_excel(path).to_dict(orient="records")
    raise ValueError(f"Unsupported result file extension: {path.suffix}")


def packet_from_result_file(
    path: str | Path,
    *,
    element: str = "hg",
    run_id: str | None = None,
    quality_threshold: float = 0.5,
) -> QCPacket:
    """Adapt one Hg result row into qc.v1 without sending raw files to an LLM."""
    result_path = Path(path)
    rows = _read_result_rows(result_path)
    if not rows:
        raise ValueError(f"Result file is empty: {result_path}")
    row = rows[-1]

    if {"element", "run_id", "metrics", "results", "flags"}.issubset(row):
        return build_qc_packet(
            element=str(row["element"]),
            run_id=str(row["run_id"]),
            metrics=dict(row["metrics"]),
            results=dict(row["results"]),
            flags=list(row["flags"]),
        )

    def first_value(*names: str) -> Any:
        for name in names:
            if name in row and row[name] not in (None, ""):
                return row[name]
        return None

    def number(value: Any) -> float | None:
        try:
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    score = number(first_value("Data_Quality_Score", "data_quality_score"))
    total_stderr = number(first_value("Total_StdErr", "total_stderr"))
    confidence = number(first_value("ML_Prediction_Confidence", "Prediction_Confidence"))
    predicted_cause = first_value("ML_Predicted_Cause", "Predicted_Cause_of_Anomaly")

    metrics: Dict[str, Any] = {}
    for output_name, source_names in {
        "data_quality_score": ("Data_Quality_Score", "data_quality_score"),
        "total_stderr": ("Total_StdErr", "total_stderr"),
        "prediction_confidence": ("ML_Prediction_Confidence", "Prediction_Confidence"),
    }.items():
        value = number(first_value(*source_names))
        if value is not None:
            metrics[output_name] = value

    result_names = {
        "delta202", "delta201", "delta200", "delta199", "R-THg(%)",
        "ML_Data_Type", "Label",
    }
    results = {name: row[name] for name in result_names if name in row and row[name] not in (None, "")}
    flags: List[Dict[str, Any]] = []
    if score is not None and score < quality_threshold:
        flags.append({
            "code": "LOW_QUALITY_SCORE",
            "severity": "critical" if score < quality_threshold / 2 else "warning",
            "message": "Data quality score is below the review threshold.",
            "source": "hg_result_adapter",
            "evidence": {"value": score, "threshold": quality_threshold},
        })
    if total_stderr is not None:
        results["total_stderr"] = total_stderr
    if predicted_cause not in (None, "", "nan"):
        flags.append({
            "code": "ML_ANOMALY_PREDICTION",
            "severity": "warning",
            "message": str(predicted_cause),
            "source": "hg_ml_prediction",
            "evidence": {"confidence": confidence},
        })

    resolved_run_id = run_id or str(first_value("run_id", "Run_ID", "Sample_ID", "Sample Name") or result_path.stem)
    return build_qc_packet(
        element=element,
        run_id=resolved_run_id,
        metrics=metrics,
        results=results,
        flags=flags,
    )


def build_llm_request(
    packet: QCPacket,
    *,
    model: str = "orcarouter/auto",
    language: str = "en",
) -> Dict[str, Any]:
    """Build an advisory-only OpenAI-compatible request body.

    Raw instrument files are intentionally excluded. Only the structured QC
    packet is sent to a provider.
    """
    return {
        "model": model,
        "temperature": 0.1,
        "response_format": {"type": "json_object"},
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a scientific QC assistant for MC-ICP-MS. "
                    "Explain evidence, rank plausible causes, and suggest checks. "
                    "Never claim certainty, never operate the instrument, and never "
                    "return an execution command. Output JSON with summary, anomalies, "
                    "recommendations, confidence."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "language": language,
                        "qc_packet": packet.to_dict(),
                        "human_confirmation_required": True,
                    },
                    ensure_ascii=False,
                ),
            },
        ],
    }


def validate_llm_assessment(raw: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate and constrain an LLM response to an advisory result."""
    summary = str(raw.get("summary", "")).strip()
    recommendations = [str(item) for item in raw.get("recommendations", [])]
    anomalies = list(raw.get("anomalies", []))
    try:
        confidence = float(raw.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0

    if not summary:
        raise ValueError("LLM assessment must include a summary")
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("LLM confidence must be between 0 and 1")

    return {
        "summary": summary,
        "anomalies": anomalies,
        "recommendations": recommendations,
        "confidence": confidence,
        "human_confirmation_required": True,
        "status": "pending_human_review",
    }


def request_llm_assessment(
    packet: QCPacket,
    *,
    model: str = "orcarouter/auto",
    language: str = "en",
    base_url: str = "https://api.orcarouter.ai/v1",
    api_key: str | None = None,
    timeout: int = 30,
) -> Dict[str, Any]:
    """Request an advisory interpretation from an OpenAI-compatible provider."""
    token = api_key or os.environ.get("ORCA_KEY")
    if not token:
        raise RuntimeError("Set ORCA_KEY or pass api_key before requesting an LLM assessment")

    body = json.dumps(build_llm_request(packet, model=model, language=language)).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=body,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"LLM provider returned HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"LLM provider request failed: {exc.reason}") from exc

    try:
        content = payload["choices"][0]["message"]["content"]
        assessment = json.loads(content) if isinstance(content, str) else content
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError("LLM provider returned an invalid structured assessment") from exc
    return validate_llm_assessment(assessment)

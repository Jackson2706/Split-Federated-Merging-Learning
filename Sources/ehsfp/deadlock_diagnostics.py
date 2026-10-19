"""Opt-in, in-process diagnostics for the E-HSFP autograd stall."""

import atexit
import faulthandler
import os
import threading
import time
from typing import Optional, TextIO

import torch


class DeadlockDiagnosticSession:
    """Own the traceback timer, CUDA sampler, and their shared log file."""

    def __init__(
        self,
        log_path: str,
        interval_seconds: float,
        anomaly_detection: bool = False,
    ):
        if interval_seconds <= 0:
            raise ValueError("diagnostic interval must be positive")
        self.log_path = log_path
        self.interval_seconds = interval_seconds
        self.anomaly_detection = anomaly_detection
        self._stop_event = threading.Event()
        self._log: Optional[TextIO] = None
        self._thread: Optional[threading.Thread] = None
        self._previous_anomaly_state = torch.is_anomaly_enabled()

    def start(self) -> "DeadlockDiagnosticSession":
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        self._log = open(self.log_path, "a", encoding="utf-8", buffering=1)
        self._write(
            f"diagnostics=start pid={os.getpid()} "
            f"interval_seconds={self.interval_seconds:g} "
            f"anomaly_detection={str(self.anomaly_detection).lower()}"
        )
        if self.anomaly_detection:
            torch.autograd.set_detect_anomaly(True)
        faulthandler.dump_traceback_later(
            self.interval_seconds,
            repeat=True,
            file=self._log,
            exit=False,
        )
        self._thread = threading.Thread(
            target=self._sample_loop,
            name="ehsfp-deadlock-diagnostics",
            daemon=True,
        )
        self._thread.start()
        return self

    def _write(self, message: str) -> None:
        if self._log is not None:
            timestamp = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            self._log.write(f"[EHSFP-DEADLOCK-DIAG] {timestamp} {message}\n")
            self._log.flush()

    def _cuda_snapshot(self) -> str:
        if not torch.cuda.is_available():
            return "cuda_available=false"
        device = torch.cuda.current_device()
        fields = [
            "cuda_available=true",
            f"device={device}",
            f"memory_allocated_bytes={torch.cuda.memory_allocated(device)}",
            f"memory_reserved_bytes={torch.cuda.memory_reserved(device)}",
        ]
        utilization = getattr(torch.cuda, "utilization", None)
        if utilization is None:
            fields.append("gpu_utilization_percent=unavailable")
        else:
            try:
                fields.append(f"gpu_utilization_percent={utilization(device)}")
            except Exception as exc:
                fields.append(
                    f"gpu_utilization_percent=error:{type(exc).__name__}"
                )
        return " ".join(fields)

    def _sample_loop(self) -> None:
        # Record a baseline immediately, then sample on the same cadence as the
        # repeated faulthandler dump. No CUDA synchronization is introduced.
        self._write(self._cuda_snapshot())
        while not self._stop_event.wait(self.interval_seconds):
            self._write(self._cuda_snapshot())

    def stop(self) -> None:
        if self._log is None:
            return
        faulthandler.cancel_dump_traceback_later()
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=min(self.interval_seconds, 1.0))
        if self.anomaly_detection:
            torch.autograd.set_detect_anomaly(self._previous_anomaly_state)
        self._write("diagnostics=stop")
        self._log.close()
        self._log = None


_active_session: Optional[DeadlockDiagnosticSession] = None


def start_deadlock_diagnostics(run_dir: str) -> DeadlockDiagnosticSession:
    """Start diagnostics after the caller has resolved its per-run directory."""
    global _active_session
    interval = float(os.environ.get("EHSFP_DEADLOCK_DIAG_SECONDS", "60"))
    anomaly_detection = (
        os.environ.get("EHSFP_DEADLOCK_DIAG_ANOMALY") == "1"
    )
    log_path = os.path.join(run_dir, "ehsfp_deadlock_diagnostics.log")
    _active_session = DeadlockDiagnosticSession(
        log_path,
        interval,
        anomaly_detection=anomaly_detection,
    ).start()
    atexit.register(_active_session.stop)
    anomaly_status = "enabled" if anomaly_detection else "disabled"
    print(
        "[EHSFP-DEADLOCK-DIAG] enabled: "
        f"tracebacks/CUDA stats every {interval:g}s; "
        f"anomaly detection {anomaly_status} -> {log_path}"
    )
    return _active_session

"""Production observability, Prometheus metrics exporter, and structured JSON logging.
Exports Prometheus metrics format without requiring external packages, enabling out-of-the-box Kubernetes monitoring."""
import json
import logging
import os
import sys
import threading
import time
from typing import Any, Dict, List, Optional


class StructuredLogger:
    """Zero-dependency structured JSON logger for cloud-native Kubernetes environments."""

    def __init__(self, name: str = "aivisibility"):
        self.logger = logging.getLogger(name)
        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            self.logger.addHandler(handler)
        self.logger.setLevel(os.environ.get("LOG_LEVEL", "INFO").upper())

    def _log(self, level: str, message: str, **kwargs: Any) -> None:
        payload = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "level": level,
            "message": message,
            **kwargs,
        }
        self.logger.info(json.dumps(payload))

    def info(self, message: str, **kwargs: Any) -> None:
        self._log("INFO", message, **kwargs)

    def warning(self, message: str, **kwargs: Any) -> None:
        self._log("WARNING", message, **kwargs)

    def error(self, message: str, **kwargs: Any) -> None:
        self._log("ERROR", message, **kwargs)


logger = StructuredLogger()


class MetricsRegistry:
    """Thread-safe, dependency-free Prometheus exposition metrics registry."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.counters: Dict[str, Dict[Tuple[str, ...], float]] = {}
        self.gauges: Dict[str, Dict[Tuple[str, ...], float]] = {}
        self.counter_help: Dict[str, str] = {}
        self.gauge_help: Dict[str, str] = {}

    def inc_counter(self, name: str, value: float = 1.0, labels: Optional[Dict[str, str]] = None, help_text: str = "") -> None:
        label_key = tuple(sorted(labels.items())) if labels else ()
        with self.lock:
            if name not in self.counters:
                self.counters[name] = {}
                self.counter_help[name] = help_text or f"Counter for {name}"
            self.counters[name][label_key] = self.counters[name].get(label_key, 0.0) + value

    def set_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None, help_text: str = "") -> None:
        label_key = tuple(sorted(labels.items())) if labels else ()
        with self.lock:
            if name not in self.gauges:
                self.gauges[name] = {}
                self.gauge_help[name] = help_text or f"Gauge for {name}"
            self.gauges[name][label_key] = value

    def to_prometheus_text(self) -> str:
        """Serializes all collected metrics into standard Prometheus exposition text format."""
        lines: List[str] = []
        with self.lock:
            # Render Counters
            for name, entries in sorted(self.counters.items()):
                lines.append(f"# HELP {name} {self.counter_help.get(name, '')}")
                lines.append(f"# TYPE {name} counter")
                for label_tuples, val in sorted(entries.items()):
                    if label_tuples:
                        lbl_str = ",".join(f'{k}="{v}"' for k, v in label_tuples)
                        lines.append(f"{name}{{{lbl_str}}} {val}")
                    else:
                        lines.append(f"{name} {val}")

            # Render Gauges
            for name, entries in sorted(self.gauges.items()):
                lines.append(f"# HELP {name} {self.gauge_help.get(name, '')}")
                lines.append(f"# TYPE {name} gauge")
                for label_tuples, val in sorted(entries.items()):
                    if label_tuples:
                        lbl_str = ",".join(f'{k}="{v}"' for k, v in label_tuples)
                        lines.append(f"{name}{{{lbl_str}}} {val}")
                    else:
                        lines.append(f"{name} {val}")

        return "\n".join(lines) + "\n"


METRICS = MetricsRegistry()

# Initialize core metrics
METRICS.inc_counter("aivisibility_requests_total", 0.0, {"endpoint": "init", "status": "200"}, "Total HTTP requests handled")
METRICS.inc_counter("aivisibility_llm_calls_total", 0.0, {"provider": "none", "model": "none"}, "Total outbound LLM API invocations")
METRICS.inc_counter("aivisibility_cache_hits_total", 0.0, help_text="Total answer cache hits")
METRICS.inc_counter("aivisibility_cache_misses_total", 0.0, help_text="Total answer cache misses")
METRICS.set_gauge("aivisibility_up", 1.0, help_text="Service operational readiness status")

"""Asynchronous background worker and task queue for distributed high-throughput audit execution.
Enables Kubernetes worker pods to scale independently from API frontends."""
import signal
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, Optional

from .ai_visibility import ask
from .db import connect, now_iso
from .observability import METRICS, logger


class AuditWorker:
    """Enterprise batch worker executing scheduled and queue-based visibility tasks."""

    def __init__(
        self,
        db_target: str,
        concurrency: int = 4,
        poll_interval: float = 2.0,
        llm_factory: Optional[Callable[[], Any]] = None,
        conn_factory: Optional[Callable[[], Any]] = None
    ):
        self.db_target = db_target
        self.concurrency = concurrency
        self.poll_interval = poll_interval
        self.llm_factory = llm_factory
        self.conn_factory = conn_factory or (lambda: connect(self.db_target, check_same_thread=False))
        self.running = False
        self._stop_event = threading.Event()

    def stop(self, *args: Any) -> None:
        """Signal the worker to gracefully shut down."""
        logger.info("AuditWorker received shutdown signal, draining in-flight tasks...")
        self.running = False
        self._stop_event.set()

    def start(self) -> None:
        """Run the worker loop, catching termination signals."""
        self.running = True
        signal.signal(signal.SIGINT, self.stop)
        signal.signal(signal.SIGTERM, self.stop)

        logger.info(
            "AuditWorker started",
            concurrency=self.concurrency,
            poll_interval=self.poll_interval,
            target=self.db_target
        )

        while self.running and not self._stop_event.is_set():
            try:
                processed = self.process_next_batch()
                if not processed:
                    self._stop_event.wait(self.poll_interval)
            except Exception as exc:
                logger.error("AuditWorker encountered error in polling cycle", error=str(exc))
                self._stop_event.wait(self.poll_interval)

        logger.info("AuditWorker shut down cleanly")

    def process_next_batch(self) -> int:
        """Fetch and execute pending queue tasks."""
        conn = self.conn_factory()
        try:
            # Check if audit_tasks table exists and has pending work
            has_table = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='audit_tasks'"
                if getattr(conn, "dialect", "sqlite") == "sqlite"
                else "SELECT 1 FROM information_schema.tables WHERE table_name='audit_tasks'"
            ).fetchone()

            if not has_table:
                return 0

            tasks = conn.execute(
                "SELECT id, title_id, model, prompt_key, prompt, sample_no, audit_run_id "
                "FROM audit_tasks WHERE status='pending' ORDER BY id ASC LIMIT ?",
                (self.concurrency,)
            ).fetchall()

            if not tasks:
                return 0

            task_ids = [t["id"] for t in tasks]
            conn.execute(
                f"UPDATE audit_tasks SET status='processing', updated_at=? WHERE id IN ({','.join('?' for _ in task_ids)})",
                [now_iso()] + task_ids
            )
            conn.commit()

            llm = self.llm_factory() if self.llm_factory else None
            if not llm:
                from .llm import get_llm
                llm = get_llm()

            with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
                futures = {
                    executor.submit(
                        self._execute_single_task,
                        dict(t),
                        llm
                    ): t["id"]
                    for t in tasks
                }
                for fut in as_completed(futures):
                    tid = futures[fut]
                    try:
                        fut.result()
                        conn.execute("UPDATE audit_tasks SET status='completed', updated_at=? WHERE id=?", (now_iso(), tid))
                    except Exception as err:
                        logger.error("Task failed", task_id=tid, error=str(err))
                        conn.execute("UPDATE audit_tasks SET status='failed', retries=retries+1, updated_at=? WHERE id=?", (now_iso(), tid))
                    conn.commit()

            return len(tasks)
        finally:
            conn.close()

    def _execute_single_task(self, task: Dict[str, Any], llm: Any) -> None:
        """Executes a single prompt inquiry with answer caching."""
        task_conn = self.conn_factory()
        try:
            ask(
                task_conn,
                llm,
                task["model"],
                task["title_id"],
                task["prompt_key"],
                task["prompt"],
                task["sample_no"],
                audit_run_id=task.get("audit_run_id")
            )
            METRICS.inc_counter("aivisibility_worker_tasks_total", 1.0, {"status": "success"})
        except Exception as e:
            METRICS.inc_counter("aivisibility_worker_tasks_total", 1.0, {"status": "error"})
            raise e
        finally:
            task_conn.close()
